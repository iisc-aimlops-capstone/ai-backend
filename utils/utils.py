import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


import os
from PIL import Image
from utils.logger import get_logger
from utils.config import load_yaml_config

config_path = root / "configs" / "config.yaml"
configs = load_yaml_config(str(config_path))

# print(configs)

# Initialize logger
logger = get_logger(__name__, log_level=configs['LOG_LEVEL'], log_file=configs['LOG_FILE_PATH'])

def load_image_from_path(image_path: str) -> Image.Image:
    """
    Load an image from a local file path.
    Args:
        image_path (str): Path to image file.
    Returns:
        PIL.Image: Loaded image.
    """
    try:
        image = Image.open(image_path)
        logger.info(f"Loaded image from {image_path}")
        return image
    except Exception as e:
        logger.exception(f"Failed to load image from {image_path}: {e}")
        raise

def load_image_from_url(image_url: str) -> Image.Image:
    """
    Download and load an image from a URL.
    Args:
        image_url (str): URL of the image.
    Returns:
        PIL.Image: Downloaded image.
    """
    import requests
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        logger.info(f"Loaded image from URL: {image_url}")
        return image
    except Exception as e:
        logger.exception(f"Failed to load image from URL {image_url}: {e}")
        raise

def get_images_from_path(image_inputs=None):
    # Sample/Multiple input images
    if image_inputs in ([], None):
        image_inputs = os.listdir(os.path.join(root, configs['INPUT_FILE_PATH']))
        logger.info(f"Loading images from local path: {os.path.join(root, 'pred')}")
        logger.info(f"Images available in the path: {image_inputs}")

    # Load images
    images = []
    for inp in image_inputs:
        if inp.startswith("http"):
            images.append(load_image_from_url(inp))
        elif os.path.exists(os.path.join(root, "pred", inp)):
            images.append(load_image_from_path(os.path.join(root, "pred", inp)))
        else:
            logger.warning(f"Invalid input, skipping: {inp}")
            return "Input Error"
    return images