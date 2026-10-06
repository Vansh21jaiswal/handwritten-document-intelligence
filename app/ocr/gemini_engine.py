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

    def predict(self, image: Image.Image) -> Tuple[str, float, str]:
        if not self.is_loaded:
            self.load_model()
            
        # Optimization: Resize image to max 1024px to drastically reduce payload size and API latency
        image.thumbnail((1024, 1024))
            
        prompt = (
            "Analyze this document image and return a JSON object with two fields:\n"
            "1. 'document_type': Classify it strictly as one of ['Handwritten Document', 'Source Code', 'Printed Document', 'ID / Form', 'General Text'].\n"
            "2. 'transcription': Transcribe ALL the text exactly as written, line by line, preserving the original reading order. "
            "Preserve code formatting if it is source code."
        )

        error_log = []
        for model_name in self.models_to_try:
            retries = 2
            for attempt in range(retries):
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        [prompt, image],
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.0,
                            response_mime_type="application/json"
                        )
                    )
                    
                    import json
                    try:
                        data = json.loads(response.text)
                        text = data.get("transcription", "").strip()
                        doc_type = data.get("document_type", "General Text")
                    except Exception:
                        text = response.text.strip()
                        doc_type = "General Text"
                    
                    return text, 0.95, doc_type
                    
                except Exception as e:
                    error_str = str(e)
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
