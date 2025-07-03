import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
print(f"Parent: {parent}")
print(f"Root: {root}")
sys.path.append(str(root))


import os
import shutil
from PIL import Image
from datetime import datetime
from typing import List, Optional
from utils.utils import get_images_from_path
from utils.logger import get_logger
from utils.config import load_yaml_config

from src.plant_disease_detection.model import PlantDiseaseClassifier

config_path = root / "configs" / "config.yaml"
configs = load_yaml_config(str(config_path))

# print(configs)

# Initialize logger
logger = get_logger(__name__, log_level=configs['LOG_LEVEL'], log_file=configs['LOG_FILE_PATH'])


def run_inference_on_images(
    images: Image.Image,  # List[Image.Image], 
    model: PlantDiseaseClassifier
    ) -> List[dict]:
    """
    Run inference on a list of images.
    Args:
        images (List[PIL.Image]): List of images to classify.
        model (PlantDiseaseClassifier): Initialized model.
    Returns:
        List[dict]: List of prediction results.
    """
    try:
        # results = []
        # if len(images)==1:
        #     results.append(model.predict_single(images[0]))
        # else:
        #     results.append(model.predict_batch(images))
        results = model.predict_single(images[0])
        logger.info("Inference completed successfully.")
        return results
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        return False

def predict_disease():
    """
    Main execution flow:
    - Load images (from user input: file paths or URLs)
    - Run inference
    - Print/present results
    """
    images, img_path = get_images_from_path()

    # Initialize model
    try:
        classifier = PlantDiseaseClassifier(os.path.join(root, configs['IMG_CLASS_MODEL_CKPT']))
        logger.info(f"Loaded model: {os.path.join(root, configs['IMG_CLASS_MODEL_CKPT'])}")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # Run inference
    results = run_inference_on_images(images, classifier)
    logger.info(results)
    if not results:
        logger.error("error")
        raise 

    shutil.rmtree(img_path)

    # # Output results
    # json_results = []
    # for idx, res in enumerate(results):
    #     logger.info("-" * 50)
    #     logger.info(f"Image {idx+1}:")
    #     logger.info(f"|-- Predicted Class: {res['predicted_class']}")
    #     logger.info(f"|-- Confidence: {res['confidence']*100:.2f}%")
    #     logger.info("-" * 50)

    #     json_results.append({
    #         "label": res['predicted_class'],
    #         "confidence": res['confidence'],
    #         "message": "Image is valid and classified successfully."
    #     })
    # return json_results
    return results

if __name__ == "__main__":
    start = datetime.now()
    predict_disease()
    end = datetime.now()
    logger.info(f"Time taken: {end-start}")