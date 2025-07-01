# validate data for wrong images and images that not included in the model.
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
# print(f"Parent: {parent}")
# print(f"Root: {root}")
sys.path.append(str(root))

import os
import yaml
from datetime import datetime

from utils.utils import get_images_from_path
from src.plant_disease_detection.data_processing import ImageProcessor
from src.plant_disease_detection.model import ImageValidator
from utils.logger import get_logger
from utils.config import load_yaml_config

config_path = root / "configs" / "config.yaml"
configs = load_yaml_config(str(config_path))

# Initialize logger
logger = get_logger(__name__)

def validate_data():
    """Main function to classify an image as a plant leaf or not."""
    try:
        # Initialize image processor and classifier
        processor = ImageProcessor()
        classifier = ImageValidator(os.path.join(root, configs["IMG_VAL_MODEL_CKPT"]))

        # Load and validate the image
        image = get_images_from_path()[0]
        if not processor.validate_image(image):
            logger.error("Image validation failed.")
            return
        logger.info("Image validation Successful.")

        # Preprocess the image
        image_tensor = processor.preprocess_image(image)

        # Perform prediction
        is_plant, label, confidence = classifier.predict(image_tensor)
        logger.info(f"Result: {label} with confidence {confidence:.2f}")

        return is_plant, label, confidence

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    start = datetime.now()
    validate_data()
    end = datetime.now()
    logger.info(f"Time taken: {end-start}")
