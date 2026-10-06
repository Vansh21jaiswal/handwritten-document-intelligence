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

# Import backend components
from app.preprocessing.validator import ImageValidator
from app.preprocessing.segmentation import ImageSegmenter
from app.extraction.structured import StructuredExtractor
from app.utils.export import generate_pdf, generate_docx, generate_json

@st.cache_resource(show_spinner="Connecting to Google Gemini Cloud...")
def load_gemini():
    from app.ocr.gemini_engine import GeminiEngine
    engine = GeminiEngine()
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
    .main .block-container { max-width: 1200px; padding-top: 2rem; }
    .hero-title { text-align: center; font-weight: 800; font-size: 2.2rem; margin-bottom: 0.5rem; }
    .hero-subtitle { text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .workflow-steps { text-align: center; font-size: 1.2rem; font-weight: bold; color: #888; margin-bottom: 2rem; }
    .workflow-active { color: #0f8243; }
    .status-high { color: #0f8243; font-weight: bold; }
    .status-low { color: #d97706; font-weight: bold; }
    .metric-card { background: var(--secondary-background-color, #f9f9f9); border-radius: 8px; padding: 15px; margin-bottom: 15px; border: 1px solid var(--border-color, #eee); }
    .metric-title { font-size: 0.85rem; color: var(--text-color, #666); opacity: 0.8; font-weight: 600; text-transform: uppercase; margin-bottom: 5px; }
    .metric-value { font-size: 1.1rem; font-weight: 700; color: var(--text-color, #111); }
    </style>
""", unsafe_allow_html=True)

# ─── TOP SECTION ────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Intelligent Document Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Transform handwritten, printed, and code images into searchable, structured text.</div>", unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Recognition")
    selected_model = st.radio(
        "OCR Engine:",
        ["AI Recognition · Gemini", "Traditional OCR (Tesseract)"],
        label_visibility="collapsed"
    )
    
    st.header("Options")
    run_validation = st.checkbox("Image Quality Check", value=True)
    run_extraction = st.checkbox("Structured Information Extraction", value=True)
    
    with st.expander("Advanced Settings"):
        run_segmentation = st.checkbox("Line Segmentation (CV)", value=True, help="Applies HPP segmentation. Visualized in Technical Analysis.")

# ─── STATE MANAGEMENT ───────────────────────────────────────────────────────
if 'page_state' not in st.session_state:
    st.session_state.page_state = "upload"
    st.session_state.results = {}

def reset_state():
    st.session_state.page_state = "upload"
    st.session_state.results = {}

# ─── WORKFLOW STEPPER ───────────────────────────────────────────────────────
stepper_placeholder = st.empty()

def update_stepper(state: str):
    if state == "upload":
        html = "<span class='workflow-active'>① Upload</span> &nbsp;─────&nbsp; <span>② Analyze</span> &nbsp;─────&nbsp; <span>③ Results</span>"
    elif state == "processing":
        html = "<span class='status-high'>✓ Upload</span> &nbsp;─────&nbsp; <span class='workflow-active'>⟳ Analyze</span> &nbsp;─────&nbsp; <span>○ Results</span>"
    else:
        html = "<span class='status-high'>✓ Upload</span> &nbsp;─────&nbsp; <span class='status-high'>✓ Analyze</span> &nbsp;─────&nbsp; <span class='workflow-active'>✓ Results</span>"
        
    stepper_placeholder.markdown(f"<div class='workflow-steps'>{html}</div>", unsafe_allow_html=True)

update_stepper(st.session_state.page_state)

# ─── MAIN APP LOGIC ─────────────────────────────────────────────────────────

if st.session_state.page_state == "upload":
    st.markdown("<div style='text-align: center; color: #666; margin-bottom: 20px;'>Supported: Handwritten Notes • Source Code • Printed Documents • Forms / IDs</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your document (Drag and drop an image here or browse files)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, width=400)
        
        if st.button("Run Recognition Pipeline", type="primary", use_container_width=True):
            st.session_state.page_state = "processing"
            update_stepper("processing")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Analyzing document...")
            image_bytes = uploaded_file.read()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            progress_bar.progress(10)
            
            if run_validation:
                status_text.text("✓ Image quality checked")
                val_res = validator.validate(image_bgr)
                st.session_state.validation = val_res
                time.sleep(0.5)
            progress_bar.progress(30)
            
            status_text.text("✓ Image preprocessing completed")
            page = segmenter.normalise_page(image_bgr)
            st.session_state.processed_image = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
            
            boxes = []
            if run_segmentation:
                boxes = segmenter.detect_lines(page)
            st.session_state.cv_boxes = boxes
            progress_bar.progress(50)
            
            status_text.text("⟳ Recognizing text...")
            t0 = time.time()
            engine = load_gemini() if "AI Recognition" in selected_model else load_tesseract()
            
            pil_img = Image.fromarray(st.session_state.processed_image)
            try:
                final_text, avg_conf, doc_type = engine.predict(pil_img)
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.session_state.page_state = "upload" # Reset state so they can try again
                if "429" in str(e) or "quota" in str(e).lower():
                    st.error(f"Google Gemini API Free-Tier Quota Exceeded. Raw error: {str(e)}")
                else:
                    st.error(f"Recognition service encountered an error: {str(e)}")
                st.stop()
                
            proc_time = time.time() - t0
            progress_bar.progress(80)
            
            status_text.text("○ Extracting structured information")
            structured_data = {}
            if run_extraction:
                structured_data = extractor.extract(final_text)
                
            progress_bar.progress(100)
            status_text.text("✓ Analysis complete")
            time.sleep(0.5)
            
            st.session_state.results = {
                "text": final_text,
                "conf": avg_conf,
                "time": proc_time,
                "doc_type": doc_type,
                "structured": structured_data,
                "model_used": "Gemini Vision" if "AI" in selected_model else "Tesseract",
                "original_image": image_bytes
            }
            st.session_state.page_state = "results"
            st.rerun()

elif st.session_state.page_state == "results":
    res = st.session_state.results
    
    col_status, col_btn = st.columns([4, 1])
    with col_status:
        st.markdown("<h3 style='color: #0f8243; margin-top: 0;'>✓ Analysis Complete</h3>", unsafe_allow_html=True)
    with col_btn:
        st.button("+ New Analysis", on_click=reset_state, use_container_width=True)
        
    st.write("---")
    
    col_main, col_side = st.columns([2, 1], gap="large")
    
    with col_side:
        # ANALYSIS SUMMARY
        st.subheader("Analysis Summary")
        
        conf_pct = int(res['conf'] * 100)
        conf_label = "High" if conf_pct > 80 else "Moderate" if conf_pct > 50 else "Low"
        
        qual_status = "Good ✓"
        if 'validation' in st.session_state and not st.session_state.validation['is_valid']:
            qual_status = "Poor ⚠"
            
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Document Type</div>
            <div class="metric-value">{res['doc_type']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Recognition Model</div>
            <div class="metric-value">{res['model_used']}</div>
        </div>
        <div class="metric-card" title="Estimated recognition confidence. This is not equivalent to measured OCR accuracy.">
            <div class="metric-title">Estimated Confidence ⓘ</div>
            <div class="metric-value">
                <span class="{'status-high' if conf_label=='High' else 'status-low'}">{conf_label}</span>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Processing Time</div>
            <div class="metric-value">{res['time']:.2f} seconds</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Image Quality</div>
            <div class="metric-value">{qual_status}</div>
        </div>
        """, unsafe_allow_html=True)

        if qual_status != "Good ✓":
            st.warning("Image quality may reduce recognition reliability.")
        
        st.write("---")
        
        # STRUCTURED INFORMATION
        if run_extraction:
            st.subheader("Structured Information")
            sd = res['structured']
            if not any(sd.values()):
                st.info("No structured information detected.")
            else:
                for k, v in sd.items():
                    if v:
                        st.markdown(f"**{k.replace('_', ' ').title()}**")
                        if isinstance(v, list):
                            for item in v:
                                st.markdown(f"- {item}")
                        elif isinstance(v, dict):
                            for dk, dv in v.items():
                                val_str = ", ".join(dv) if isinstance(dv, list) else dv
                                st.markdown(f"- {dk}: {val_str}")
                        else:
                            st.markdown(f"- {v}")
                            
            with st.expander("View JSON"):
                st.write("Machine-readable JSON payload:")
                json_data = generate_json(res['text'], {
                    "doc_type": res['doc_type'],
                    "model": res['model_used'],
                    "time": round(res['time'], 2),
                    "confidence_label": conf_label
                }, sd)
                st.code(json_data.decode('utf-8'), language="json")

    with col_main:
        # RECOGNIZED TEXT
        st.subheader("Recognized Text")
        
        if res['doc_type'] == "Source Code":
            st.code(res['text'], language="python")
        else:
            if not res['text'].strip():
                st.info("No recognizable text detected in the image.")
            else:
                # Using st.code with text language gives an adaptive container WITH a native copy button!
                st.code(res['text'], language="text")
            
        st.write("---")
        
        # EXPORT SYSTEM
        st.subheader("Export Results")
        doc_col, data_col = st.columns(2)
        
        metadata = {
            "doc_type": res['doc_type'],
            "model": res['model_used'],
            "time": round(res['time'], 2),
            "confidence_label": conf_label
        }
        
        doc_col.write("**Document Exports**")
        doc_col.download_button("📥 Download PDF", data=generate_pdf(res['text'], metadata, res.get('structured', {})), file_name="report.pdf", mime="application/pdf")
        doc_col.download_button("📥 Download DOCX", data=generate_docx(res['text'], metadata), file_name="document.docx")
        doc_col.download_button("📥 Download TXT", data=res['text'], file_name="extraction.txt")
        
        data_col.write("**Data Exports**")
        data_col.download_button("📥 Download JSON", data=generate_json(res['text'], metadata, res.get('structured', {})), file_name="data.json", mime="application/json")
        # Only meaningful CSV logic (key-values) could go here if requested, but JSON is prioritized.
        
        st.write("---")
        
        # CV PREPROCESSING VISUALIZATION
        with st.expander("Computer Vision & Preprocessing Visualization"):
            tab1, tab2 = st.tabs(["Original Image", "Processed Image (CV)"])
            with tab1:
                st.image(res['original_image'], use_container_width=True)
            with tab2:
                if 'processed_image' in st.session_state:
                    disp_img = st.session_state.processed_image.copy()
                    if run_segmentation and 'cv_boxes' in st.session_state:
                        for (y1, y2, x1, x2) in st.session_state.cv_boxes:
                            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    st.image(disp_img, use_container_width=True, caption="OpenCV normalisation and HPP Line detection bounds.")
                else:
                    st.info("Preprocessing was disabled.")

        # TECHNICAL ANALYSIS
        with st.expander("Technical Analysis"):
            st.write(f"**Recognition Model:** {res['model_used']}")
            st.write(f"**Processing Time:** {res['time']:.2f} seconds")
            st.write(f"**Preprocessing:** {'Enabled' if run_validation else 'Disabled'}")
            st.write(f"**Line Segmentation:** {'Enabled' if run_segmentation else 'Disabled'}")
            if run_segmentation and 'cv_boxes' in st.session_state:
                st.write(f"**Detected Lines:** {len(st.session_state.cv_boxes)}")
            st.write(f"**Structured Extraction:** {'Enabled' if run_extraction else 'Disabled'}")
            if 'validation' in st.session_state:
                m = st.session_state.validation['metrics']
                st.write(f"**Input Resolution:** {m.get('width')} × {m.get('height')}")
                st.write(f"**Laplacian Blur Score:** {m.get('blur_score')}")
                st.write(f"**Contrast STD Score:** {m.get('contrast_score')}")

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### How It Works")
st.markdown("""
<div style='background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center; color: #334155; font-size: 0.95rem; font-weight: 500;'>
    Upload &nbsp; ➔ &nbsp; Quality Check &nbsp; ➔ &nbsp; CV Preprocessing &nbsp; ➔ &nbsp; AI Recognition &nbsp; ➔ &nbsp; Entity Extraction &nbsp; ➔ &nbsp; Export
</div>
<br><div style='color:#666; font-size:0.9rem; text-align: center;'>
The system combines computer-vision preprocessing, AI/OCR-based recognition, post-processing, and structured information extraction to convert document images into usable digital data.
</div>
""", unsafe_allow_html=True)
st.markdown("<br><div style='color:#999; font-size:0.8rem; text-align: center;'><i>Privacy Note: Uploaded documents are processed in-memory for recognition and are not intentionally stored by this application.</i></div>", unsafe_allow_html=True)
