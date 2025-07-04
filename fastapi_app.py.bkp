import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
print(f"Parent: {parent}")
sys.path.append(str(parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
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

config_path = parent / "configs" / "config.yaml"
configs = load_yaml_config(str(config_path))

# Initialize logger
logger = get_logger(__name__, log_level=configs['LOG_LEVEL'], log_file=configs['LOG_FILE_PATH'])

os.makedirs(configs['INPUT_FILE_PATH'], exist_ok=True)

class ValidationResult(BaseModel):
    """Response model for image validation and classification results."""
    filename: str
    image: str
    is_plant: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    message: str

@app.post("/validate_and_classify/", response_model=List[ValidationResult], summary="Validate images")
async def validate_and_classify_images(files: List[UploadFile] = File(..., description="List of image files to validate")):
    """
    Validate a list of images in parallel using Dask.

    Args:
        files (List[UploadFile]): List of image files to validate.

    Returns:
        List[ValidationResult]: List of validation results for each image.
    """
    try:
        # Extract file paths from uploaded files
        image_path = [file.filename for file in files][0]
        logger.info(image_path)

        results = []
        for file in files:
            # Save the uploaded file to the upload folder
            file_path = os.path.join(configs['INPUT_FILE_PATH'], file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(file.file.read())

            is_plant, label, is_plant_confidence = validate_data()

            if not is_plant:
                return {
                    "filename": file.filename,
                    "image": "xxx",
                    "is_plant": str(is_plant),
                    "label": None,
                    "confidence": None,
                    "message": "Image validation failed."
                }
                continue

            prediction_results  = predict_disease()
            # Append the results for this image
            results.append({
                "filename": file.filename,
                "image": "xxx",
                "is_plant": f"{is_plant} with confidence: {is_plant_confidence}",
                "label": prediction_results['predicted_class'],  # Assuming predict_disease returns a list
                "confidence": prediction_results['confidence'],
                "message": "Image is valid and classified successfully."
            })
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/health/", summary="Check API health")
async def health_check():
    """
    Check the health of the API.

    Returns:
        dict: Health status of the API.
    """
    return {"status": "healthy"}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
