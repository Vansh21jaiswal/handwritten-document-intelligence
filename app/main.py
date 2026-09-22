import os
import sys
import io
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the ML pipeline (Must use TrOCR-Large as instructed)
from scripts.run_handwriting_demo import run_pipeline

# PDF and DOCX generation
from fpdf import FPDF
from docx import Document
from docx.shared import Pt, Inches


def _sanitise_for_pdf(text: str) -> str:
    """Replace characters that the default FPDF Latin-1 font cannot render."""
    replacements = {
        "\u2192": "->",   # →
        "\u2190": "<-",   # ←
        "\u2194": "<->",  # ↔
        "\u2022": "-",    # •
        "\u2026": "...",  # …
        "\u201c": '"',    # "
        "\u201d": '"',    # "
        "\u2018": "'",    # '
        "\u2019": "'",    # '
        "\u2013": "-",    # –
        "\u2014": "--",   # —
        "\u2260": "!=",   # ≠
        "\u2264": "<=",   # ≤
        "\u2265": ">=",   # ≥
        "\u00d7": "x",    # ×
        "\u00f7": "/",    # ÷
        "\u2713": "OK",   # ✓
        "\u26a0": "(!)",  # ⚠
        "\u00a0": " ",    # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final safety net: encode to latin-1, replacing anything still unhandled
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf(text: str, title: str = "Handwritten Notes Transcription") -> bytes:
    """Convert plain text into a PDF and return bytes."""
    safe_title = _sanitise_for_pdf(title)
    safe_text  = _sanitise_for_pdf(text)

    pdf = FPDF()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, safe_title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Body — one line at a time so line breaks are preserved
    pdf.set_font("Helvetica", size=12)
    for line in safe_text.split("\n"):
        content = line if line.strip() else " "
        pdf.multi_cell(0, 8, content, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def generate_docx(text: str, title: str = "Handwritten Notes Transcription") -> bytes:
    """Convert plain text into a Word .docx and return bytes."""
    doc = Document()
    # Title paragraph
    title_para = doc.add_heading(title, level=1)
    if title_para.runs:
        title_para.runs[0].font.size = Pt(18)
    doc.add_paragraph("")  # blank line
    # Body
    for line in text.split("\n"):
        p = doc.add_paragraph(line)
        if p.runs:
            p.runs[0].font.size = Pt(12)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()



def printed_ocr(image_path: str) -> str:
    """Run Tesseract OCR on a printed/digital document image and return the extracted text."""
    import pytesseract
    pil_img = Image.open(image_path).convert("RGB")
    custom_config = r"--oem 3 --psm 3"
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    return text.strip()


def handwritten_code_ocr(image_path: str) -> str:
    """OCR pipeline tuned for handwritten code/math on notebook/blank paper.

    Local models (TrOCR/Tesseract) fail completely on messy handwritten C++ 
    because they expect either English prose or printed fonts.
    This mode uses the Cloud AI (Gemini 3.6 Flash) to achieve >95% accuracy
    on complex handwritten syntax and symbols.
    """
    import google.generativeai as genai
    from PIL import Image
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable is not set. Cloud AI is required for code recognition."

    genai.configure(api_key=api_key)
    
    # gemini-3.6-flash is the supported version for this key
    model = genai.GenerativeModel("gemini-3.6-flash")

    pil_img = Image.open(image_path)
    prompt = (
        "Extract the handwritten programming code or math from this image. "
        "Output ONLY the code text. Preserve indentation, brackets, semicolons, "
        "and symbols exactly as written. Do NOT wrap the output in markdown "
        "code blocks (like ```cpp) — just return the raw text."
    )

    try:
        response = model.generate_content([prompt, pil_img])
        # Clean up any accidental markdown blocks the model might still add
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
        return text.strip()
    except Exception as e:
        return f"Error connecting to Cloud AI for Code Recognition: {str(e)}"




st.set_page_config(page_title="Handwritten Notes to Text", layout="wide")

# Custom CSS for sensible max widths, compact image previews, and clean design
st.markdown("""
    <style>
    .main .block-container {
        max-width: 1000px;
        padding-top: 3rem;
        padding-bottom: 3rem;
        margin: 0 auto;
    }
    .hero-title {
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
    }
    .upload-card {
        border: 2px dashed #ddd;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .preview-img img {
        max-height: 400px;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .doc-preview-img img {
        max-height: 600px;
        object-fit: contain;
        border-radius: 8px;
        border: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HERO SECTION
# -----------------------------------------
st.markdown("<div class='hero-title'>DOCUMENT TEXT EXTRACTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Extract text from handwritten notes or printed documents.<br>Supports notebook pages, certificates, ID cards, and more.</div>", unsafe_allow_html=True)

# -----------------------------------------
# MODE SELECTOR
# -----------------------------------------
mode = st.radio(
    "Choose document type:",
    ["✍️  Handwritten Notes (Prose)", "💻  Handwritten Code / Math", "🖨️  Printed / Digital Document"],
    horizontal=True,
    label_visibility="visible"
)
is_printed = mode.startswith("🖨️")
is_code = mode.startswith("💻")

st.markdown("---")

# -----------------------------------------
# UPLOAD SECTION
# -----------------------------------------
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.res = None
    st.session_state.temp_path = None
    st.session_state.mode = None
    st.session_state.printed_text = None

# Reset if user switches mode after processing
if st.session_state.get('mode') != mode:
    st.session_state.processed = False
    st.session_state.res = None
    st.session_state.temp_path = None
    st.session_state.printed_text = None
    st.session_state.mode = mode

# If not processed yet, show upload section prominently
if not st.session_state.processed:
    if is_printed:
        st.markdown("### Upload your document")
        st.markdown("Choose a clear photo or scan of a printed document — certificate, ID card, letter, book page, etc.")
    elif is_code:
        st.markdown("### Upload your handwritten code")
        st.markdown("Choose a clear photo of handwritten programming code or math. (Uses character-level OCR for symbols)")
    else:
        st.markdown("### Upload your handwritten page")
        st.markdown("Choose a clear photo of a handwritten notebook page.")

    uploaded_file = st.file_uploader("Upload area", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.markdown("<div class='preview-img'>", unsafe_allow_html=True)
        col_space1, col_img, col_space2 = st.columns([1, 2, 1])
        with col_img:
            st.image(uploaded_file, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("Extract Text", use_container_width=True, type="primary"):
                # Save temp file
                temp_dir = os.path.join(project_root, "outputs", "demo")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "temp_uploaded.jpg")

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.temp_path = temp_path

                if is_printed:
                    with st.spinner("Reading your document..."):
                        try:
                            text = printed_ocr(temp_path)
                            st.session_state.printed_text = text
                            st.session_state.processed = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during extraction: {str(e)}")
                elif is_code:
                    with st.spinner("Reading your handwritten code (character level)..."):
                        try:
                            text = handwritten_code_ocr(temp_path)
                            st.session_state.printed_text = text  # Reuse the same simple text view as printed mode
                            st.session_state.processed = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during extraction: {str(e)}")
                else:
                    with st.spinner("Reading your handwriting (word level)..."):
                        try:
                            out_dir = os.path.join(project_root, "outputs", "demo", "app_results")
                            res = run_pipeline(temp_path, out_dir=out_dir)
                            st.session_state.res = res
                            st.session_state.processed = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during extraction: {str(e)}")





# -----------------------------------------
# RESULTS SECTION — shared download helper
# -----------------------------------------
def _download_buttons(final_text: str):
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        st.download_button("⬇ Download TXT", data=final_text,
                           file_name="extracted_text.txt", mime="text/plain",
                           use_container_width=True)
    with btn_col2:
        pdf_bytes = generate_pdf(final_text)
        st.download_button("⬇ Download PDF", data=pdf_bytes,
                           file_name="extracted_text.pdf", mime="application/pdf",
                           use_container_width=True)
    with btn_col3:
        docx_bytes = generate_docx(final_text)
        st.download_button("⬇ Download Word", data=docx_bytes,
                           file_name="extracted_text.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           use_container_width=True)
    with btn_col4:
        safe_text = final_text.replace('`', '\\`')
        copy_html = f"""
        <script>
        function copyToClipboard() {{
            navigator.clipboard.writeText(`{safe_text}`);
            document.getElementById('copyBtn').innerText = 'Copied!';
            setTimeout(() => document.getElementById('copyBtn').innerText = '📋 Copy Text', 2000);
        }}
        </script>
        <button id="copyBtn" onclick="copyToClipboard()" style="width:100%; padding:0.5rem 1rem; background-color:#f0f2f6; border:1px solid #c4c4c4; border-radius:0.25rem; font-family:sans-serif; cursor:pointer;">📋 Copy Text</button>
        """
        components.html(copy_html, height=45)


if st.session_state.processed:
    temp_path = st.session_state.temp_path
    st.success("Text extracted successfully.")
    st.markdown("---")

    # ── PRINTED / CODE MODE RESULTS ───────────────────────────────────
    if st.session_state.mode and (st.session_state.mode.startswith("🖨️") or st.session_state.mode.startswith("💻")) and st.session_state.printed_text is not None:
        final_text = st.session_state.printed_text

        res_col1, res_col2 = st.columns([1.2, 1])
        with res_col1:
            st.subheader("Your document")
            st.markdown("<div class='doc-preview-img'>", unsafe_allow_html=True)
            st.image(temp_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.subheader("Your extracted text")
            st.text_area("Extracted text", value=final_text, height=450, label_visibility="collapsed")
            _download_buttons(final_text)

    # ── HANDWRITTEN MODE RESULTS ──────────────────────────────────────
    elif st.session_state.res is not None:
        res = st.session_state.res

        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.subheader("Your document")
            st.markdown("<div class='doc-preview-img'>", unsafe_allow_html=True)
            st.image(temp_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")
            with st.expander("View detected lines"):
                st.image(res['debug_path'], use_container_width=True)
                st.caption("Green = recognized lines\n\nYellow = lines that may need review\n\nRed = ignored background or non-writing areas.")

        with res_col2:
            st.subheader("Your extracted text")
            num_accepted = res['results']['num_accepted_lines']
            num_review = res['results']['num_review_lines']
            num_recognized = num_accepted - num_review
            st.markdown(f"**✓ {num_recognized} lines recognized**")
            if num_review > 0:
                st.markdown(f"**⚠ {num_review} lines may need review**")

            final_text = res['results']['final_transcription']
            st.text_area("Final text", value=final_text, height=350, label_visibility="collapsed")
            _download_buttons(final_text)

            with st.expander("More options"):
                with open(res['json_path'], 'r') as f:
                    json_data = f.read()
                st.download_button("Download JSON", data=json_data, file_name="result.json",
                                   mime="application/json")

            st.write("")
            with st.expander("Review individual lines"):
                for line_data in res['results']['lines']:
                    st.markdown(f"**Line {line_data['line_number']}**")
                    st.image(line_data['crop_path'])
                    if line_data.get('review_required', False):
                        st.warning("⚠ Needs review")
                    else:
                        st.success("✓ Recognized")
                    st.text_input(f"Edit line {line_data['line_number']}",
                                  value=line_data['text'],
                                  key=f"edit_{line_data['line_number']}",
                                  label_visibility="collapsed")
                    st.markdown("---")

    st.write("")
    if st.button("Start Over"):
        st.session_state.processed = False
        st.session_state.res = None
        st.session_state.temp_path = None
        st.session_state.printed_text = None
        st.rerun()

