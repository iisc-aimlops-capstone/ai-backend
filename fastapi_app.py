import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
print(f"Parent: {parent}")
sys.path.append(str(parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import boto3
import asyncio
import uuid
import io
import datetime
from botocore.exceptions import ClientError, NoCredentialsError
import tempfile
from src.plant_disease_detection.data_validation import validate_data
from src.plant_disease_detection.infer import predict_disease
from src.plant_disease_detection.rag_system import RagApp
from src.plant_disease_detection.translate import translate_text
from utils.logger import get_logger
from utils.config import load_yaml_config
import openai
from googletrans import Translator, LANGUAGES
import google.generativeai as genai
import json
from PIL import Image


# Load environment variables
# Uncomment the following line if you are using a .env file 
from dotenv import load_dotenv
load_dotenv()

# --- Dependency Injection for Translator ---
# This makes your app easier to test and manage.
def get_translator():
    return Translator()

# Set the OpenAI API in Environment
# openai.api_key = os.environ.get("OPENAI_API_KEY")

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-exp')


# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="An API to Identify plant diseases and provide suitable recommendation.",
    version="1.0.0",
    root_path="/api"
)

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config_path = parent / "configs" / "config.yaml"
configs = load_yaml_config(str(config_path))

# Initialize logger
logger = get_logger(__name__, log_level=configs['LOG_LEVEL'], log_file=configs['LOG_FILE_PATH'])

# Create directories
os.makedirs(configs['INPUT_FILE_PATH'], exist_ok=True)

# Initialize S3 client
S3_REGION = os.environ.get("AWS_REGION", "us-east-2")
s3_client = boto3.client("s3", region_name=S3_REGION)
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "s3b-iisc-aimlops-cap-images")

class S3ImageRequest(BaseModel):
    """Request model for S3 image analysis."""
    file_key: str

class ValidationResult(BaseModel):
    """Response model for image validation and classification results."""
    filename: str
    image: str
    is_plant: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    message: str
    disease_details: Optional[str] = None


def download_from_s3(bucket_name: str, file_key: str, local_path: str) -> bool:
    """
    Download a file from S3 to local storage.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        file_key (str): Key of the file in S3
        local_path (str): Local path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        s3_client.download_file(bucket_name, file_key, local_path)
        logger.info(f"Successfully downloaded {file_key} from S3 bucket {bucket_name}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error(f"File {file_key} does not exist in bucket {bucket_name}")
        elif error_code == 'NoSuchBucket':
            logger.error(f"Bucket {bucket_name} does not exist")
        else:
            logger.error(f"ClientError downloading from S3: {e}")
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading from S3: {e}")
        return False

def delete_from_s3(bucket_name: str, file_key: str) -> bool:
    """
    Delete a file from S3.
    Args:
        bucket_name (str): Name of the S3 bucket
        file_key (str): Key of the file in S3
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=file_key)
        logger.info(f"Successfully deleted {file_key} from S3 bucket {bucket_name}")
        return True
    except ClientError as e:
        logger.error(f"ClientError deleting from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting from S3: {e}")
        return False

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str

class ChatWithImageRequest(BaseModel):
    message: str
    image_base64: str
    conversation_history: Optional[List[dict]] = []

# System prompt for plant disease chatbot
PLANT_DISEASE_SYSTEM_PROMPT = """
You are PlantCare AI, an expert plant disease diagnosis and treatment assistant. Your expertise includes:

1. **Plant Disease Identification**: Accurately identify plant diseases from symptoms and images
2. **Treatment Recommendations**: Provide specific, actionable treatment advice
3. **Prevention Strategies**: Suggest preventive measures for plant health
4. **Organic Solutions**: Prioritize eco-friendly and organic treatment methods
5. **Regional Context**: Consider local climate and farming practices
6. **Safety Guidelines**: Always emphasize safe handling of treatments

Guidelines:
- Always maintain a helpful, professional, and encouraging tone
- Provide detailed, step-by-step instructions when giving treatment advice
- Ask clarifying questions when symptoms are unclear
- Recommend consulting local agricultural experts for severe cases
- Include timing information for treatments (when to apply, frequency)
- Suggest monitoring and follow-up actions
- Prioritize sustainable and environmentally friendly solutions

Response Format:
- Be concise but comprehensive
- Use bullet points for treatment steps
- Include warnings about chemical treatments
- Provide alternative organic options when possible
- Give realistic timelines for recovery

Remember: You are here to help farmers and gardeners maintain healthy plants through expert guidance and practical solutions.
"""

# In-memory conversation storage (in production, use a database)
conversations = {}


@app.post("/analyze_from_s3/", response_model=ValidationResult, summary="Analyze image from S3")
async def analyze_image_from_s3(request: S3ImageRequest):
    """
    Download an image from S3 and analyze it for plant disease detection.
    
    Args:
        request (S3ImageRequest): Request containing S3 file key
        
    Returns:
        ValidationResult: Analysis results for the image
    """
    disease_details=None
    try:
        logger.info(f"Processing image: {request.file_key} from bucket: {S3_BUCKET_NAME}")
        
        # Create a temporary file to store the downloaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(request.file_key)[1]) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Download image from S3 using the configured bucket name
            if not download_from_s3(S3_BUCKET_NAME, request.file_key, temp_file_path):
                raise HTTPException(
                    status_code=404, 
                    detail=f"Failed to download image {request.file_key} from S3 bucket {S3_BUCKET_NAME}"
                )
            
            # Copy the downloaded file to the input directory for processing
            local_file_path = os.path.join(configs['INPUT_FILE_PATH'], request.file_key)
            
            # Copy temp file to input directory
            import shutil
            shutil.copy2(temp_file_path, local_file_path)
            
            logger.info(f"Image saved locally at: {local_file_path}")
            
            # Validate if the image contains a plant
            try:
                is_plant, label, is_plant_confidence, img_path = validate_data()
                logger.info(f"Plant validation result: {is_plant}, confidence: {is_plant_confidence}")
            except Exception as e:
                logger.error(f"Error in plant validation: {e}")
                raise HTTPException(status_code=500, detail=f"Plant validation failed: {str(e)}")
            
            if not is_plant:
                return ValidationResult(
                    filename=request.file_key,
                    image="processed",
                    is_plant=f"False with confidence: {is_plant_confidence}",
                    label=None,
                    confidence=None,
                    message="Image validation failed. The uploaded image does not appear to contain a plant.",
                    disease_details=None
                )
            
            # Predict disease if it's a plant
            try:
                prediction_results = predict_disease()
                logger.info(f"Disease prediction results: {prediction_results}")
                rag_retreival = RagApp()
                question = f"Provide all information about the disease {prediction_results['predicted_class']}"
                disease_details = rag_retreival.run(question)
                logger.info(disease_details)
                
                if 'generation' not in disease_details:
                    logger.info("No response generated")
                else:
                    disease_details = disease_details["generation"]
            except Exception as e:
                logger.error(f"Error in disease prediction: {e}")
                raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")
            
            return ValidationResult(
                filename=request.file_key,
                image="processed",
                is_plant=f"True with confidence: {is_plant_confidence}",
                label=prediction_results.get('predicted_class', 'Unknown'),
                confidence=prediction_results.get('confidence', 0.0),
                message="Image is valid and classified successfully.",
                disease_details=disease_details
            )
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                # Optionally clean up the local file as well
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
            # Delete from S3 after processing
            delete_from_s3(S3_BUCKET_NAME, request.file_key)
            # Ensure pred folder exists after cleanup
            os.makedirs(configs['INPUT_FILE_PATH'], exist_ok=True)
                
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_image_from_s3: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/validate_and_classify/", response_model=List[ValidationResult], summary="Validate images (original endpoint)")
async def validate_and_classify_images(files: List[UploadFile] = File(..., description="List of image files to validate")):
    """
    Original endpoint - Validate a list of uploaded images.
    
    Args:
        files (List[UploadFile]): List of image files to validate.
        
    Returns:
        List[ValidationResult]: List of validation results for each image.
    """
    disease_details = None
    try:
        results = []
        for file in files:
            # Save the uploaded file to the upload folder
            if not os.path.exists(os.path.join(parent, configs['INPUT_FILE_PATH'])):
                os.makedirs(os.path.join(parent, configs['INPUT_FILE_PATH']))
            file_path = os.path.join(parent, configs['INPUT_FILE_PATH'], file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())

            try:
                is_plant, label, is_plant_confidence, img_path = validate_data()
            except Exception as e:
                logger.error(f"Error in plant validation: {e}")
                results.append(ValidationResult(
                    filename=file.filename,
                    image="error",
                    is_plant="Error",
                    label=None,
                    confidence=None,
                    message=f"Plant validation failed: {str(e)}",
                    disease_details=None
                ))
                continue

            if not is_plant:
                if os.path.isfile(img_path):
                    os.remove(img_path)
                elif os.path.isdir(img_path):
                    # Only remove files inside, not the folder itself
                    for f in os.listdir(img_path):
                        file_path = os.path.join(img_path, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                results.append(ValidationResult(
                    filename=file.filename,
                    image="processed",
                    is_plant=f"False with confidence: {is_plant_confidence}",
                    label=None,
                    confidence=None,
                    message="Image validation failed. The uploaded image does not appear to contain a plant.",
                    disease_details=None
                ))
                continue

            try:
                prediction_results = predict_disease()
                if 'healthy' in prediction_results['predicted_class'].lower():
                    results.append(ValidationResult(
                        filename=file.filename,
                        image="processed",
                        is_plant=f"True with confidence: {is_plant_confidence}",
                        label=prediction_results.get('predicted_class', 'None'),
                        confidence=prediction_results.get('confidence', 0.0),
                        message="Plant is Healthy. No additional information is needed.",
                        disease_details="Healthy Plant"
                    ))
                    return results
                rag_retreival = RagApp()
                question = f"Provide all information about the disease {prediction_results['predicted_class']}"
                disease_details = rag_retreival.run(question)
                logger.info(disease_details)
                
                if 'generation' not in disease_details:
                    logger.info("No response generated")
                    results.append(ValidationResult(
                        filename=file.filename,
                        image="error",
                        is_plant=f"True with confidence: {is_plant_confidence}",
                        label=prediction_results.get('predicted_class', 'None'),
                        confidence=prediction_results.get('confidence', 0.0),
                        message=f"Disease information not yet available in out database. We are constantly working to update our records and will have this information soon.",
                        disease_details="Disease information not available yet."
                    ))
                    # return results
                
                    logger.info("No response generated from custom model, trying Gemini fallback")
                    
                    # Gemini fallback integration
                    try:
                        # Call Gemini for disease identification
                        gemini_classification = await classify_image_with_gemini(file_path)
                        
                        if gemini_classification:
                            # Parse the Gemini response
                            gemini_result = json.loads(gemini_classification)
                            
                            # Get disease details from Gemini
                            gemini_disease_details = await get_disease_details_with_gemini(gemini_result)
                            
                            results.append(ValidationResult(
                                filename=file.filename,
                                image="processed",
                                is_plant=gemini_result.get('is_plant', f"True with confidence: {is_plant_confidence}"),
                                label=gemini_result.get('label', 'Unknown'),
                                confidence=gemini_result.get('confidence', 0.0),
                                message=gemini_result.get('message', "[Results generated from Gemini-2.5-flash] Image is valid and classified successfully."),
                                disease_details=gemini_disease_details
                            ))
                        else:
                            # Both custom model and Gemini failed
                            results.append(ValidationResult(
                                filename=file.filename,
                                image="error",
                                is_plant=f"True with confidence: {is_plant_confidence}",
                                label=prediction_results.get('predicted_class', 'None'),
                                confidence=prediction_results.get('confidence', 0.0),
                                message="Disease prediction failed: Both custom model and Gemini fallback failed",
                                disease_details=None
                            ))
                    except Exception as gemini_error:
                        logger.error(f"Gemini fallback failed: {gemini_error}")
                        results.append(ValidationResult(
                            filename=file.filename,
                            image="error",
                            is_plant=f"True with confidence: {is_plant_confidence}",
                            label=prediction_results.get('predicted_class', 'None'),
                            confidence=prediction_results.get('confidence', 0.0),
                            message=f"Disease prediction failed: Custom model failed, Gemini fallback error: {str(gemini_error)}",
                            disease_details=None
                        ))
                    return results
                else:
                    # Custom model succeeded
                    disease_details = disease_details["generation"]
                    results.append(ValidationResult(
                        filename=file.filename,
                        image="processed",
                        is_plant=f"True with confidence: {is_plant_confidence}",
                        label=prediction_results.get('predicted_class', 'Unknown'),
                        confidence=prediction_results.get('confidence', 0.0),
                        message="Image is valid and classified successfully.",
                        disease_details=disease_details
                    ))
                    
            except Exception as e:
                logger.error(f"Error in disease prediction: {e}")
                
                # Try Gemini as fallback when custom model fails completely
                try:
                    logger.info("Custom model failed completely, trying Gemini fallback")
                    gemini_classification = await classify_image_with_gemini(file_path)
                    
                    if gemini_classification:
                        gemini_result = json.loads(gemini_classification)
                        gemini_disease_details = await get_disease_details_with_gemini(gemini_result)
                        
                        results.append(ValidationResult(
                            filename=file.filename,
                            image="processed",
                            is_plant=gemini_result.get('is_plant', f"True with confidence: {is_plant_confidence}"),
                            label=gemini_result.get('label', 'Unknown'),
                            confidence=gemini_result.get('confidence', 0.0),
                            message=gemini_result.get('message', "[Results generated from Gemini-2.5-flash] Image is valid and classified successfully."),
                            disease_details=gemini_disease_details
                        ))
                    else:
                        results.append(ValidationResult(
                            filename=file.filename,
                            image="error",
                            is_plant=f"True with confidence: {is_plant_confidence}",
                            label=None,
                            confidence=0.0,
                            message=f"Disease prediction failed: {str(e)}",
                            disease_details=None
                        ))
                except Exception as gemini_error:
                    logger.error(f"Gemini fallback failed: {gemini_error}")
                    results.append(ValidationResult(
                        filename=file.filename,
                        image="error",
                        is_plant=f"True with confidence: {is_plant_confidence}",
                        label=None,
                        confidence=0.0,
                        message=f"Disease prediction failed: {str(e)}",
                        disease_details=None
                    ))
                continue
                
        os.makedirs(configs['INPUT_FILE_PATH'], exist_ok=True)
        return results

    except Exception as e:
        logger.error(f"Unexpected error in validate_and_classify_images: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# Helper functions for Gemini integration
async def classify_image_with_gemini(image_path: str) -> str:
    """
    Classify image using Gemini model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: JSON string with classification results
    """
    try:
        # Initialize Gemini model (adjust based on your Gemini setup)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Read and prepare the image
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Prepare the image for Gemini
        image = {
            'mime_type': 'image/jpeg',  # or 'image/png' based on your image
            'data': image_data
        }
        
        # Get the prompt
        prompt = get_disease_identification_prompt()
        
        # Generate response
        response = model.generate_content([prompt, image])
        
        # Clean the response to ensure it's valid JSON
        response_text = response.text.strip()
        
        # Remove any markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        return response_text
        
    except Exception as e:
        logger.error(f"Error in Gemini classification: {e}")
        return None


async def get_disease_details_with_gemini(classification_result: dict) -> str:
    """
    Get detailed disease information using Gemini text model.
    
    Args:
        classification_result (dict): The classification result from Gemini
        
    Returns:
        str: Detailed disease information
    """
    try:
        # Initialize Gemini model for text generation
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Get the prompt for disease details
        prompt = get_disease_details_prompt(classification_result)
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error in Gemini disease details: {e}")
        return "Unable to generate disease details at this time."


@app.post("/chat/text", response_model=ChatResponse)
async def chat_text_only(request: ChatMessage):
    """Handle text-only chat messages"""
    try:
        # Create conversation history context
        context = PLANT_DISEASE_SYSTEM_PROMPT + "\n\nConversation History:\n"
        
        for msg in request.conversation_history[-5:]:  # Last 5 messages for context
            context += f"User: {msg.get('user', '')}\n"
            context += f"Assistant: {msg.get('assistant', '')}\n"
        
        context += f"\nCurrent User Question: {request.message}\n"
        context += "\nPlease provide a helpful response about plant care, disease diagnosis, or treatment:"
        
        # Generate response using Gemini
        response = model.generate_content(context)
        
        # Generate conversation ID and timestamp
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Store conversation (in production, use database)
        conversations[conversation_id] = {
            "messages": request.conversation_history + [
                {"user": request.message, "assistant": response.text}
            ],
            "timestamp": timestamp
        }
        
        return ChatResponse(
            response=response.text,
            conversation_id=conversation_id,
            timestamp=timestamp
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@app.post("/chat/image")
async def chat_with_image(
    message: str = Form(...),
    image: UploadFile = File(...),
    conversation_history: str = Form("[]")
):
    """Handle chat with image upload"""
    try:
        # Parse conversation history
        try:
            conv_history = json.loads(conversation_history)
        except:
            conv_history = []
        
        # Read and process image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Create context with conversation history
        context = PLANT_DISEASE_SYSTEM_PROMPT + "\n\nConversation History:\n"
        
        for msg in conv_history[-5:]:  # Last 5 messages for context
            context += f"User: {msg.get('user', '')}\n"
            context += f"Assistant: {msg.get('assistant', '')}\n"
        
        context += f"\nCurrent User Question: {message}\n"
        context += """
Please analyze the uploaded plant image and provide:
1. Plant identification (if possible)
2. Disease/problem diagnosis
3. Severity assessment
4. Detailed treatment recommendations
5. Prevention strategies
6. Expected recovery timeline

Focus on practical, actionable advice that the user can implement immediately.
"""
        
        # Generate response with image
        response = model.generate_content([context, pil_image])
        
        # Generate conversation ID and timestamp
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Store conversation
        conversations[conversation_id] = {
            "messages": conv_history + [
                {
                    "user": message,
                    "assistant": response.text,
                    "has_image": True,
                    "image_name": image.filename
                }
            ],
            "timestamp": timestamp
        }
        
        return {
            "response": response.text,
            "conversation_id": conversation_id,
            "timestamp": timestamp,
            "image_processed": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image chat processing error: {str(e)}")

@app.get("/chat/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Retrieve conversation history"""
    try:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversations[conversation_id]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.delete("/chat/history/{conversation_id}")
async def clear_conversation_history(conversation_id: str):
    """Clear specific conversation history"""
    try:
        if conversation_id in conversations:
            del conversations[conversation_id]
            return {"message": "Conversation history cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.get("/chat/health")
async def chat_health_check():
    """Health check for chat functionality"""
    try:
        # Test Gemini connection
        test_response = model.generate_content("Hello, this is a test message for plant care.")
        return {
            "status": "healthy",
            "gemini_connection": "active",
            "active_conversations": len(conversations),
            "test_response_length": len(test_response.text)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "gemini_connection": "failed"
        }
    
@app.get("/health/", summary="Check API health")
async def health_check():
    """
    Check the health of the API.
    
    Returns:
        dict: Health status of the API.
    """
    return {
        "status": "healthy",
        "message": "Plant Disease Detection API is running",
        "version": "1.0.0"
    }

# --- Pydantic Models for Clear Contracts ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str


@app.post("/translate")
async def translate_text(
    request: TranslationRequest,
    translator: Translator = Depends(get_translator)
):
    """
    Translates the given text to a target language asynchronously.
    """
    if request.target_language not in LANGUAGES:
        logger.warning(f"Invalid language code received: {request.target_language}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target language code '{request.target_language}'. Please use a valid code from the supported list."
        )
    try:
        # Run the potentially slow I/O operation in the background
        translated = await translator.translate(
            text=request.text,
            dest=request.target_language
        )
        # translated = translator.translate(
        #     request.text,
        #     dest=request.target_language
        # )
        logger.info(f"Successfully translated text to '{request.target_language}'")
        # return TranslationResponse(translated_text=translated.text)
        return {"translated_text": translated.text}

    except Exception as e:
        logger.error(f"Translation failed for target language '{request.target_language}': {e}", exc_info=True)
        return {"error": f"Translation failed: {str(e)}"}
        # raise HTTPException(
        #     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        #     detail="An unexpected error occurred during translation. The issue has been logged."
        # )



# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)