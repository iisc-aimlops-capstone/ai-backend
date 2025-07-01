import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
# print(f"Parent: {parent}")
# print(f"Root: {root}")
sys.path.append(str(root))

from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from utils.logger import get_logger
from utils.config import load_yaml_config

config_path = root / "configs" / "config.yaml"
configs = load_yaml_config(str(config_path))
preprocessing_config = configs['image_preprocessing']

logger = get_logger(__name__, log_level=configs['LOG_LEVEL'], log_file=configs['LOG_FILE_PATH'])


class ImageProcessor:
    """Handles image loading, preprocessing, and validation."""

    def __init__(self):
        # MobileNet-compatible preprocessing
        self.mobilenet_transform = transforms.Compose([
            transforms.Resize(preprocessing_config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing_config["mean"],
                std=preprocessing_config["std"]
            )
        ])
        logger.info("ImageProcessor initialized with MobileNet preprocessing.")

    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from a file path and convert it to RGB.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            PIL.Image: Loaded image in RGB mode.
        
        Raises:
            UnidentifiedImageError: If the image cannot be identified or opened.
            FileNotFoundError: If the image file does not exist.
            OSError: If the image is corrupted or cannot be read.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Image loaded successfully from {image_path}.")
            return image
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def validate_image(self, image: Image.Image) -> bool:
        """Validate the image to ensure it is not empty and is in RGB mode.
        
        Args:
            image (PIL.Image): Image to validate.
        
        Returns:
            bool: True if the image is valid, False otherwise.
        """
        if image.size == (0, 0):
            logger.warning("Image is empty or corrupted.")
            return False
        if image.mode != "RGB":
            logger.warning("Image is not in RGB mode.")
            return False
        logger.info("Image is valid.")
        return True

    def preprocess_image(self, image: Image.Image):
        """Preprocess an image to a tensor suitable for MobileNet.
        
        Args:
            image (PIL.Image): Image to preprocess.
        
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        tensor = self.mobilenet_transform(image).unsqueeze(0)
        logger.info("Image preprocessed successfully.")
        return tensor