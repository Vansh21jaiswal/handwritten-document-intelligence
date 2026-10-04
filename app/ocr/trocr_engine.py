import torch
from PIL import Image
from typing import Tuple, Dict, Any
from app.ocr.base import OCRModel

class TrOCREngine(OCRModel):
    """
    Deep Learning OCR using Microsoft's TrOCR (Transformer-based Optical Character Recognition).
    Excellent for handwriting.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.is_loaded = False

    def load_model(self) -> None:
        if self.is_loaded:
            return
            
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        # use_fast=False bypasses Streamlit Cloud tokenizer serialization issues
        self.processor = TrOCRProcessor.from_pretrained(self.model_name, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        if not self.is_loaded:
            self.load_model()
            
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=3,               # Speed/accuracy balance
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        ids = outputs.sequences
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        
        # TrOCR sequences_scores represents the log probability of the generated sequence
        conf_score = 1.0
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            # Convert log probability to a 0.0 - 1.0 confidence score
            conf_score = torch.exp(outputs.sequences_scores[0]).item()
            
        return text, float(conf_score)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "type": "Deep Learning (VisionEncoderDecoder)",
            "backend": "PyTorch + HuggingFace Transformers",
            "device": str(self.device),
            "strengths": "Highly accurate on varied handwriting styles",
            "weaknesses": "Compute heavy, slow without GPU"
        }
