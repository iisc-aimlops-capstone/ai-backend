def get_template_1(context, question):
    return f"""
        You are an expert assistant specializing in plant diseases. You have received raw text from a database to answer a user's question about a disease that was identified from an image.

        Your primary task is to first clean the raw text to isolate the main article, and then use that article to answer the user's specific question.

        Instructions:

        1. Isolate the Main Article: Read the entire Raw Text Context. Ignore all surrounding boilerplate content, such as navigation menus (e.g., "HOME", "SEARCH"), page headers, and footers (e.g., "Statewide IPM Program", "Copyright", "Legal Notices"), to identify the core article about the plant disease.

        2. Analyze the Article for Headings: Within the main article you have isolated, find the existing headings (e.g., "Identification", "Life cycle", "Damage", "Solutions").

        3. Answer the User's Question: Use the information from the cleaned article to answer the user's Question.

        4. Format Your Answer:

            - Structure the entire response using Markdown.

            - Use the exact headings you found in the article (e.g., ### Identification, ### Solutions) to organize your answer.

            - Use bullet points (*) for the details under each heading.

        5. Handle Missing Information: If the cleaned article does not contain the information needed to answer the question, or if it is missing key sections like "Symptoms" or "Treatment", you must respond only with the following message, without any  additional/hallucinated information using other information:

            - "I'm sorry, but detailed information for your query is not yet available in our database. We are constantly working to update our records and will have this information available soon."

        Raw Text Context:
        {context}

        Question:
        {question}

        Your Formatted Answer:
        """


def get_disease_identification_prompt() -> str:
    """
    Returns the prompt for the multimodal model to classify the image.
    This prompt strictly enforces a JSON-only response.
    """
    return (
        "Analyze the provided image of a plant leaf. You must perform the following tasks:\n"
        "1.  Determine if the image contains a plant.\n"
        "2.  Identify the specific plant disease. If the plant is healthy, the predicted class should be 'healthy'.\n"
        "3.  Provide your confidence level for the disease prediction as a float between 0.0 and 1.0.\n\n"
        "Your response MUST be a valid JSON object and nothing else. Do not add any text, explanations, or markdown formatting before or after the JSON. The JSON object must have these exact keys:\n"
        "- `is_plant`: {True/False} with confidence: {your confidence level}.\n"
        "- `label`: A string with the disease name (e.g., 'Tomato_Early_Blight') or 'healthy {plant name}'.\n"
        "- `confidence`: A float representing the prediction confidence (e.g., 0.97).\n\n"
        "- `message`: Mention the image is validated and generated from Gemini-2.5-flash.\n\n"
        "Example response for a diseased leaf:\n"
        '{\n'
        '  "is_plant": True with confidence: 0.98,\n'
        '  "image": "processed",\n'
        '  "label": "Tomato_Early_Blight",\n'
        '  "confidence": 0.97,\n'
        '  "message": [Results generated from Gemini-2.5-flash] Image is valid and classified successfully.,\n'
        '}'
        "Example response for a healthy leaf:\n"
        '{\n'
        '  "is_plant": True with confidence: 0.72,\n'
        '  "image": "processed",\n'
        '  "label": "Healthy {plant name}",\n'
        '  "confidence": 0.97,\n'
        '  "message": [Results generated from Gemini-2.5-flash] Image is valid and classified successfully.,\n'
        '}'
        "Example response for not plant/leaf image:\n"
        '{\n'
        '  "is_plant": False with confidence: 0.99,\n'
        '  "image": "processed",\n'
        '  "label": None,\n'
        '  "confidence": None,\n'
        '  "message": [Results generated from Gemini-2.5-flash] Image validation failed. The uploaded image does not appear to contain a plant..,\n'
        '}'
    )


def get_disease_details_prompt(disease_name: str) -> str:
    """
    Returns the prompt for the text model to generate disease details.
    It takes the classification result as input to provide context.
    """
    return (
        f"Act as a plant science expert. A plant has been classified with the disease '{disease_name}'.\n\n"
        "Provide a detailed guide for this specific condition. The guide must include the following sections:\n"
        "- **Overview**: A brief description of what '{disease_name}' is and what typically causes it (e.g., fungus, bacteria, virus).\n"
        "- **Symptoms**: A bulleted list of key visual signs to look for on the plant's leaves, stems, or fruit.\n"
        "- **Solution/Treatment**: An actionable, bulleted list of methods to manage and treat the disease. If applicable, include both organic and chemical options.\n"
        "- **Prevention**: A bulleted list of proactive measures to prevent future outbreaks.\n\n"
        "Your response should only contain this guide. Do not add any conversational introductions or conclusions."
    )