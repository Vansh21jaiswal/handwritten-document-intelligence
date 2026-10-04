from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import time

from app.preprocessing.validator import ImageValidator
from app.preprocessing.segmentation import ImageSegmenter
from app.extraction.structured import StructuredExtractor
# Note: In a true production environment, models would be loaded into memory on startup.
# For simplicity here, we initialize them when needed or keep them at module level.

app = FastAPI(
    title="Handwritten Document Intelligence API",
    description="API for recognizing handwritten and printed documents.",
    version="2.0.0"
)

# Global components
validator = ImageValidator()
segmenter = ImageSegmenter()
extractor = StructuredExtractor()

# Lazy loaded models
models = {}

def get_model(model_name: str):
    if model_name not in models:
        if model_name == "tesseract":
            from app.ocr.tesseract_engine import TesseractEngine
            models[model_name] = TesseractEngine()
        elif model_name == "trocr":
            from app.ocr.trocr_engine import TrOCREngine
            models[model_name] = TrOCREngine()
        else:
            raise ValueError("Unknown model requested.")
        models[model_name].load_model()
    return models[model_name]


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "available_models": ["tesseract", "trocr"]}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("trocr"),
    run_segmentation: bool = Form(True)
):
    """
    Main prediction endpoint.
    1. Validates the image.
    2. Segments lines (if run_segmentation=True).
    3. Runs OCR via the requested engine.
    4. Extracts structured data.
    """
    t0 = time.time()
    
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload JPG or PNG.")

    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    # 1. Validation
    validation = validator.validate(image_bgr)
    if not validation["is_valid"]:
        return JSONResponse(status_code=422, content={
            "error": "Image Quality Check Failed",
            "reason": validation["reason"],
            "metrics": validation["metrics"]
        })

    # 2. Setup Model
    try:
        engine = get_model(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    lines_data = []
    full_text = ""
    avg_confidence = 0.0

    # 3. Processing & OCR
    if run_segmentation and model == "trocr":
        # Complex pipeline for handwriting
        page = segmenter.normalise_page(image_bgr)
        boxes = segmenter.detect_lines(page)
        
        confidences = []
        for i, (y1, y2, x1, x2) in enumerate(boxes):
            crop_bgr = page[y1:y2, x1:x2]
            status, reason = segmenter.categorize_crop(crop_bgr)
            
            if status in ["accepted", "uncertain"]:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                
                text, conf = engine.predict(pil_crop)
                
                lines_data.append({
                    "line_number": len(lines_data) + 1,
                    "text": text,
                    "confidence": round(conf, 4),
                    "status": status,
                    "bbox": [int(y1), int(y2), int(x1), int(x2)]
                })
                confidences.append(conf)
                
        full_text = "\n".join([line["text"] for line in lines_data])
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
    else:
        # Simple pipeline (Tesseract handles full page natively)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        full_text, avg_confidence = engine.predict(pil_img)

    # 4. Structured Extraction
    structured_data = extractor.extract(full_text)

    processing_time = round(time.time() - t0, 3)

    return {
        "status": "success",
        "model_used": engine.get_model_info(),
        "processing_time_ms": int(processing_time * 1000),
        "overall_confidence": round(avg_confidence, 4),
        "validation_metrics": validation["metrics"],
        "text": full_text,
        "structured_data": structured_data,
        "lines": lines_data
    }
