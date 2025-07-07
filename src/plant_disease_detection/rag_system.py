import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
print(f"Parent: {parent}")
print(f"Root: {root}")
sys.path.append(str(root))


import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, TypedDict

from utils.logger import get_logger
from utils.prompts import get_template_1
from utils.config import load_yaml_config

config_path = root / "configs" / "config.yaml"
configs = load_yaml_config(str(config_path))

load_dotenv()

# Initialize logger
logger = get_logger(__name__, log_level=configs['LOG_LEVEL'], log_file=configs['LOG_FILE_PATH'])


from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI

# --- LangGraph Components ---
from langgraph.graph import StateGraph, END


# --- Define the LangGraph State and Nodes ---
class GraphState(TypedDict):
    question: str
    documents: List[Document] # Documents are now LangChain Document objects
    generation: str


# --- Document Processing and Vector Store Creation ---
def create_vector_store(df, embeddings, chroma_db):
    """Loads documents from a DataFrame, formats them, and creates a ChromaDB vector store."""
    print("Creating LangChain Documents from DataFrame...")

    documents = [
        Document(
            page_content=f"Plant Crop_host: {row['plant_crop_host']}\nCommon Name: {row['common_name']}\nScientific name: {row['scientific_name']}\nType: {row['Type']}\nDisease Information: {row['disease_information']}",
            metadata={
                "common_name": row['common_name'],
                "plant_crop_host": row['plant_crop_host'],
                "type": row['Type']
            }
        )
        for index, row in df.iterrows()
    ]
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_db
    )
    print("Vector store created successfully.")

    return vectorstore


# --- Step 5: Define the RAG Workflow as a Class ---
class RagApp:
    def __init__(self, persist_directory=os.path.join(root, configs['CHROMA_DB'])):
        """Initializes the retriever, LLM, and the compiled LangGraph app."""
        print("Initializing RagApp...")
        self.persist_directory = persist_directory
        # 1. Initialize Tools
        # 2. Check for and load/create the vector store
        # self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from '{self.persist_directory}'...")
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print(f"No existing vector store found. Creating a new one at '{self.persist_directory}'...")
            data_df = load_data()
            vector_store = create_vector_store(data_df, self.embeddings)

        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.5},
            )  # "score_threshold": 0.8, "fetch_k": 3, ,  "lambda_mult": 1 , "k": 3

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # gpt-4o-mini  gpt-3.5-turbo
        
        # 2. Build and Compile the Graph
        self.app = self._build_graph()
        print("RagApp initialized successfully.")

    def _build_graph(self):
        """Builds and compiles the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("generate", END)
        workflow.add_conditional_edges(
            "retrieve",
            self.decide_to_generate,
            {"generate": "generate", "end": END},
        )
        return workflow.compile()

    # --- Graph Node Definitions ---
    def retrieve_documents(self, state):
        """Node to retrieve documents from the vector store."""
        print("---NODE: RETRIEVE DOCUMENTS---")
        question = state["question"]
        # The retriever is now accessed via self.retriever
        documents = self.retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents.")
        return {"documents": documents, "question": question}

    def generate_answer(self, state):
        """Node to generate an answer using the OpenAI LLM."""
        print("---NODE: GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        def format_full_context(document):
            meta = document.metadata
            lines = [
                f"Plant/Crop Host: {meta.get('plant_crop_host', '')}",
                f"Common Name: {meta.get('common_name', '')}",
                f"Scientific Name: {meta.get('scientific_name', '')}",
                f"Type: {meta.get('type', '')}",
                f"Disease Information: {meta.get('disease_information', '')}"
            ]
            return "\n".join(lines)
        all_contexts = "\n\n".join([format_full_context(doc) for doc in documents])
        template = get_template_1(all_contexts, question)
        # template = f"""
        # You are an expert assistant specializing in plant diseases. You have received raw text from a database to answer a user's question about a disease that was identified from an image.

        # Your primary task is to first clean the raw text to isolate the main article, and then use that article to answer the user's specific question.

        # Instructions:

        # 1. Isolate the Main Article: Read the entire Raw Text Context. Ignore all surrounding boilerplate content, such as navigation menus (e.g., "HOME", "SEARCH"), page headers, and footers (e.g., "Statewide IPM Program", "Copyright", "Legal Notices"), to identify the core article about the plant disease.

        # 2. Analyze the Article for Headings: Within the main article you have isolated, find the existing headings (e.g., "Identification", "Life cycle", "Damage", "Solutions").

        # 3. Answer the User's Question: Use the information from the cleaned article to answer the user's Question.

        # 4. Format Your Answer:

        #     - Structure the entire response using Markdown.

        #     - Use the exact headings you found in the article (e.g., ### Identification, ### Solutions) to organize your answer.

        #     - Use bullet points (*) for the details under each heading.

        # 5. Handle Missing Information: If the cleaned article does not contain the information needed to answer the question, or if it is missing key sections like "Symptoms" or "Treatment", you must respond only with the following message, without any  additional/hallucinated information using other information:

        #     - "I'm sorry, but detailed information for your query is not yet available in our database. We are constantly working to update our records and will have this information available soon."

        # Raw Text Context:
        # {context}

        # Question:
        # {question}

        # Your Formatted Answer:
        # """
        prompt = ChatPromptTemplate.from_template(template)
        
        # The LLM is now accessed via self.llm
        rag_chain = prompt | self.llm | StrOutputParser()
        
        generation = rag_chain.invoke({"context": all_contexts, "question": question})
        print("Generated answer from OpenAI.")
        # print(generation)
        return {"documents": documents, "question": question, "generation": generation}

    def decide_to_generate(self, state):
        """Conditional edge: Decides whether to generate an answer or end."""
        print("---NODE: DECIDE TO GENERATE---")
        if not state["documents"]:
            print("Decision: No documents found. Ending graph.")
            return "end"
        else:
            print(f"Decision: {len(state['documents'])} Documents found. Proceeding to generate.")
            print(state["documents"])
            return "generate"

    def run(self, question: str):
        """Runs the RAG application with a given question."""
        inputs = {"question": question}
        return self.app.invoke(inputs)