import os
import time
import google.generativeai as genai
from PIL import Image
from typing import Tuple, Dict, Any
from app.ocr.base import OCRModel

class GeminiEngine(OCRModel):
    """
    Cloud AI Engine using Google's Gemini.
    Outsources the heavy deep learning compute to Google, avoiding Streamlit CPU throttling.
    """

    def __init__(self):
        self.is_loaded = False
        self.api_key = os.environ.get("GEMINI_API_KEY")
        
        # We use the specific 3.6 model that your key is authorized for
        self.models_to_try = [
            "gemini-3.6-flash",
            "gemini-1.5-flash"
        ]

    def load_model(self) -> None:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=self.api_key)
        self.is_loaded = True

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        if not self.is_loaded:
            self.load_model()
            
        prompt = (
            "This is a photograph of a handwritten notebook page or document. "
            "Transcribe ALL the text exactly as written, line by line, "
            "preserving the original reading order from top to bottom. "
            "Output ONLY the transcribed text. Do NOT add any markdown formatting."
        )

        error_log = []
        for model_name in self.models_to_try:
            retries = 2
            for attempt in range(retries):
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        [prompt, image],
                        generation_config=genai.types.GenerationConfig(temperature=0.0)
                    )
                    text = response.text.strip()
                    
                    if text.startswith("```"):
                        lines = text.split("\n")
                        if len(lines) > 2:
                            text = "\n".join(lines[1:-1]).strip()
                    
                    # Gemini doesn't return exact confidence for text generation, so we default to high (0.95)
                    # if the API succeeds, as Gemini's OCR capability is near state-of-the-art.
                    return text, 0.95
                    
                except Exception as e:
                    error_str = str(e)
                    # Handle the strict 5-request-per-minute Free Tier limit silently
                    if "429" in error_str and "minute" in error_str.lower() and attempt < retries - 1:
                        time.sleep(25)
                        continue
                    
                    error_log.append(f"[{model_name}] {error_str}")
                    break
                    
        raise RuntimeError(f"Gemini API failed: {' | '.join(error_log)}")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Gemini 3.6 Flash (Cloud API)",
            "type": "Multimodal LLM",
            "backend": "Google Generative AI",
            "strengths": "Instant inference, avoids CPU throttling, high accuracy",
            "weaknesses": "Requires internet, strict 5-requests/minute free quota"
        }
