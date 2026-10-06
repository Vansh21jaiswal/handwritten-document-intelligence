from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from PIL import Image

class OCRModel(ABC):
    """
    Abstract base class for all OCR models.
    Ensures a consistent interface whether using traditional OCR (Tesseract)
    or Deep Learning OCR (TrOCR).
    """

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory. Should be idempotent."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Tuple[str, float, str]:
        """
        Run inference on a single image/crop.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (recognized_text, confidence_score [0.0 to 1.0], document_type)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the model (name, type, architecture)."""
        pass
