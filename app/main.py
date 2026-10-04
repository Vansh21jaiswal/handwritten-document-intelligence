import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Ensure imports work regardless of execution directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import our new decoupled backend components
from app.preprocessing.validator import ImageValidator
from app.preprocessing.segmentation import ImageSegmenter
from app.extraction.structured import StructuredExtractor

# Caching the OCR models so they load instantly on subsequent requests
@st.cache_resource(show_spinner="Connecting to Google Gemini Cloud...")
def load_gemini():
    from app.ocr.gemini_engine import GeminiEngine
    engine = GeminiEngine()
    # If on Streamlit Cloud, it will pull from st.secrets if environment variable isn't set
    if not os.environ.get("GEMINI_API_KEY") and "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    engine.load_model()
    return engine

@st.cache_resource(show_spinner="Initializing Traditional OCR...")
def load_tesseract():
    from app.ocr.tesseract_engine import TesseractEngine
    engine = TesseractEngine()
    engine.load_model()
    return engine

# Initialize global components
validator = ImageValidator()
segmenter = ImageSegmenter()
extractor = StructuredExtractor()

st.set_page_config(page_title="Intelligent Document Recognition", layout="wide")

# Custom CSS for polished UI
st.markdown("""
    <style>
    .main .block-container { max-width: 1100px; padding-top: 2rem; }
    .hero-title { text-align: center; font-weight: 800; font-size: 2.2rem; margin-bottom: 0.5rem; }
    .hero-subtitle { text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem; }
    .status-high { color: #0f8243; font-weight: bold; }
    .status-low { color: #d97706; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero-title'>Intelligent Handwritten Document Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>End-to-end AIML pipeline demonstrating CV preprocessing, Cloud AI OCR, and structured data extraction.</div>", unsafe_allow_html=True)

# ─── Sidebar Config ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Configuration")
    
    selected_model = st.radio(
        "Select OCR Engine",
        ["Cloud AI (Gemini Vision)", "Traditional Baseline (Tesseract)"],
        help="Compare performance between Traditional CV and Cloud AI."
    )
    
    st.markdown("---")
    st.markdown("**Image Preprocessing**")
    run_validation = st.checkbox("Run Quality Validation", value=True)
    run_segmentation = st.checkbox("Run Line Segmentation (HPP)", value=True, help="Crucial for handwriting. Can disable for clean printed text.")
    
    st.markdown("---")
    st.markdown("**Structured Extraction**")
    run_extraction = st.checkbox("Extract JSON Entities", value=True)


# ─── Main Upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Document (Handwritten Notes, Forms, Printed Docs)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load Image
    image_bytes = uploaded_file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    col_img, col_results = st.columns([1, 1.2], gap="large")
    
    with col_img:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # 2. Image Validation
        if run_validation:
            with st.expander("Image Quality Metrics", expanded=True):
                val_res = validator.validate(image_bgr)
                m = val_res["metrics"]
                st.write(f"**Resolution:** {m.get('width', 0)} x {m.get('height', 0)}")
                st.write(f"**Blur Score (Laplacian):** {m.get('blur_score', 0)}")
                st.write(f"**Contrast Score (STD):** {m.get('contrast_score', 0)}")
                
                if not val_res["is_valid"]:
                    st.error(f"Validation Failed: {val_res['reason']}")
                    st.stop()
                else:
                    st.success("Image quality is acceptable.")

    with col_results:
        if st.button("Run Recognition Pipeline", type="primary", use_container_width=True):
            
            with st.spinner("Processing pipeline..."):
                t0 = time.time()
                
                engine = load_gemini() if "Cloud AI" in selected_model else load_tesseract()
                
                final_text = ""
                avg_conf = 0.0
                lines_data = []
                
                # 3. Preprocessing & OCR
                # TrOCR requires line segmentation. Gemini and Tesseract natively process full pages.
                if run_segmentation and "TrOCR" in selected_model:
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
                            lines_data.append({"text": text, "conf": conf, "status": status})
                            confidences.append(conf)
                            
                    final_text = "\n".join([l["text"] for l in lines_data])
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                else:
                    # Full page inference (Gemini / Tesseract)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(image_rgb)
                    final_text, avg_conf = engine.predict(pil_img)
                
                proc_time = time.time() - t0
                
            # ─── Display Results ──────────────────────────────────────────────
            st.subheader("Recognition Results")
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Processing Time", f"{proc_time:.2f} s")
            
            # Confidence formatting
            conf_pct = int(avg_conf * 100)
            if conf_pct > 80:
                conf_str = f"<span class='status-high'>{conf_pct}% (High)</span>"
            elif conf_pct > 50:
                conf_str = f"<span class='status-low'>{conf_pct}% (Moderate)</span>"
            else:
                conf_str = f"<span style='color:red; font-weight:bold;'>{conf_pct}% (Low)</span>"
                st.warning("Low confidence prediction. Result may require manual verification.")
                
            m2.markdown(f"**Confidence:**<br>{conf_str}", unsafe_allow_html=True)
            m3.metric("Model", "Gemini Cloud" if "Cloud AI" in selected_model else "Tesseract")

            # Raw Text
            st.text_area("Extracted Text", value=final_text, height=250)
            
            # 4. Structured Extraction
            if run_extraction:
                st.subheader("Structured Extraction")
                structured_data = extractor.extract(final_text)
                if not any(structured_data.values()):
                    st.info("No structured entities (Dates, Amounts, Phones, Emails) detected.")
                else:
                    st.json(structured_data)
                    
            # Export
            st.download_button("Download Text", data=final_text, file_name="extraction.txt", use_container_width=True)
