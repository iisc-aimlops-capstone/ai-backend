import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
# print(f"Parent: {parent}")
# print(f"Root: {root}")
sys.path.append(str(root))


from transformers import AutoImageProcessor, AutoModelForImageClassification, MobileNetV2ImageProcessor
import torch
import torch.nn as nn
from torchvision import models
from typing import List, Union
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class ImageValidator:
    """A binary classifier to determine if an image is a plant leaf or not."""

    def __init__(self, model_path: str, device: str = None):
        """Initialize the classifier with the given model path and device.
        
        Args:
            model_path (str): Path to the model weights file.
            device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        # Set device to GPU if available, otherwise CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load the model
        self.model = self._load_model(model_path)
        logger.info(f"Loaded model: {model_path}")

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the binary leaf classifier model.
        
        Args:
            model_path (str): Path to the model weights file.
        
        Returns:
            torch.nn.Module: The loaded MobileNetV2 model for binary classification.
        """
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()

    def predict(self, image_tensor: torch.Tensor) -> tuple[str, float]:
        """Classify an image as a plant leaf or not.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor.
        
        Returns:
            tuple: (label, confidence) where label is "plant_leaf" or "not_plant_leaf",
                and confidence is the prediction confidence score.
        
        Raises:
            RuntimeError: If inference fails.
        """
        try:
            with torch.no_grad():
                logits = self.model(image_tensor.to(self.device))
                pred = logits.argmax(1).item()
                conf = torch.softmax(logits, dim=1)[0][pred].item()

            is_plant = True if pred == 1 else False
            label = "plant_leaf" if pred == 1 else "not_plant_leaf"
            logger.info(f"Prediction: {label} with confidence {conf:.2f}")
            return is_plant, pred, conf
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")


class PlantDiseaseClassifier:
    def __init__(self, model_ckpt: str):
        """
        Initializes the image classification model and processor.
        Args:
            model_ckpt (str): Pretrained model checkpint/identifier.
        """
        try:
            # self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
            self.processor = MobileNetV2ImageProcessor.from_pretrained(model_ckpt)
            self.model = AutoModelForImageClassification.from_pretrained(model_ckpt)
            logger.info(f"Model from '{model_ckpt}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from '{model_ckpt}': {e}")
            raise

    def predict_single(self, image):
        """
        Predict class for a single image.
        Args:
            image (PIL.Image): Input image.
        Returns:
            dict: {'predicted_class': str, 'confidence': float}
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred_idx = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][pred_idx].item()
            label = self.model.config.id2label[pred_idx]
            return {'predicted_class': label, 'confidence': confidence}
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, images: List, parallel: bool = True) -> List[dict]:
        """
        Predict classes for a batch of images. Supports parallel processing.
        Args:
            images (List[PIL.Image]): List of images.
            parallel (bool): If True, use parallel processing.
        Returns:
            List[dict]: List of prediction results.
        """
        if parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(self.predict_single, images))
            return results
        else:
            return [self.predict_single(img) for img in images]