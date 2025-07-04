import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
print(f"Parent: {parent}")
sys.path.append(str(parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import tempfile
from src.plant_disease_detection.data_validation import validate_data
from src.plant_disease_detection.infer import predict_disease
from utils.logger import get_logger
from utils.config import load_yaml_config

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="An API to Identify plant diseases and provide suitable recommendation.",
    version="1.0.0",
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


@app.post("/analyze_from_s3/", response_model=ValidationResult, summary="Analyze image from S3")
async def analyze_image_from_s3(request: S3ImageRequest):
    """
    Download an image from S3 and analyze it for plant disease detection.
    
    Args:
        request (S3ImageRequest): Request containing S3 file key
        
    Returns:
        ValidationResult: Analysis results for the image
    """
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
                    message="Image validation failed. The uploaded image does not appear to contain a plant."
                )
            
            # Predict disease if it's a plant
            try:
                prediction_results = predict_disease()
                logger.info(f"Disease prediction results: {prediction_results}")
            except Exception as e:
                logger.error(f"Error in disease prediction: {e}")
                raise HTTPException(status_code=500, detail=f"Disease prediction failed: {str(e)}")
            
            return ValidationResult(
                filename=request.file_key,
                image="processed",
                is_plant=f"True with confidence: {is_plant_confidence}",
                label=prediction_results.get('predicted_class', 'Unknown'),
                confidence=prediction_results.get('confidence', 0.0),
                message="Image is valid and classified successfully."
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
                    message=f"Plant validation failed: {str(e)}"
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
                    message="Image validation failed. The uploaded image does not appear to contain a plant."
                ))
                continue

            try:
                prediction_results = predict_disease()
            except Exception as e:
                logger.error(f"Error in disease prediction: {e}")
                results.append(ValidationResult(
                    filename=file.filename,
                    image="error",
                    is_plant=f"True with confidence: {is_plant_confidence}",
                    label=None,
                    confidence=None,
                    message=f"Disease prediction failed: {str(e)}"
                ))
                continue
                
            # Append the results for this image
            results.append(ValidationResult(
                filename=file.filename,
                image="processed",
                is_plant=f"True with confidence: {is_plant_confidence}",
                label=prediction_results.get('predicted_class', 'Unknown'),
                confidence=prediction_results.get('confidence', 0.0),
                message="Image is valid and classified successfully."
            ))
        os.makedirs(configs['INPUT_FILE_PATH'], exist_ok=True)
        return results

    except Exception as e:
        logger.error(f"Unexpected error in validate_and_classify_images: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

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

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)