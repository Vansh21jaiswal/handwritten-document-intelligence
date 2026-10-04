import pytesseract
from PIL import Image
from typing import Tuple, Dict, Any
from app.ocr.base import OCRModel

class TesseractEngine(OCRModel):
    """
    Traditional OCR Baseline using Tesseract.
    Excellent for printed text, but struggles significantly with handwriting.
    """

    def __init__(self, config: str = "--oem 3 --psm 3"):
        self.config = config
        self.is_loaded = False

    def load_model(self) -> None:
        """Tesseract does not require a heavy model load into RAM."""
        self.is_loaded = True

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Run Tesseract OCR.
        Tesseract can provide confidence scores if we use image_to_data, 
        but for full page string extraction, we estimate confidence based on 
        data output, or fallback to a baseline confidence.
        """
        # To get confidence, we use image_to_data
        data = pytesseract.image_to_data(image, config=self.config, output_type=pytesseract.Output.DICT)
        
        text_parts = []
        confidences = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = data['conf'][i]
            
            if text and conf != '-1':
                text_parts.append(text)
                confidences.append(float(conf) / 100.0) # Tesseract returns 0-100
                
        final_text = " ".join(text_parts)
        
        # Aggregate confidence (average word confidence)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return final_text.strip(), avg_conf

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Tesseract OCR",
            "type": "Traditional Baseline",
            "backend": "pytesseract",
            "strengths": "Fast, excellent on printed text",
            "weaknesses": "Poor on cursive/handwriting"
        }
