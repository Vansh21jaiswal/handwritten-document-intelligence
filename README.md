---
title: Handwritten Document Intelligence
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: app/main.py
pinned: false
---

# Handwritten Document Intelligence

Convert handwritten notebook pages, printed documents, and handwritten code into editable digital text.

## Overview

This application provides an AI-assisted end-to-end pipeline for recognizing text from images. It handles three primary types of documents:
1. **Handwritten Notes (Prose)**: English handwritten notebook pages.
2. **Handwritten Code / Math**: Complex handwritten programming syntax and math expressions.
3. **Printed / Digital Documents**: Clean digital text from certificates, ID cards, and printed pages.

*Note: Handwritten text recognition is inherently challenging. While the system uses state-of-the-art models, difficult cursive, ambiguous symbols, and messy handwriting may still require manual review. The application provides an interface to review and correct low-confidence lines.*

**🚀 No API keys required — all AI models run locally on the server.**

## Main Workflow

1. **Upload handwritten document**
   User uploads a photo of a notebook page or document.
2. **Detect handwriting lines**
   The system automatically deskews the image and applies line segmentation.
3. **Filter invalid/background regions**
   Non-text regions like desk edges, blank space, and ruled lines are filtered out.
4. **Recognize handwriting**
   The cropped lines are passed through the recognition model.
5. **Ordered transcription**
   The lines are reconstructed into paragraph order.
6. **Export text**
   The user can copy the text or download it as TXT, PDF, or Word (DOCX).

## Current Models

All models run locally — no cloud APIs, no rate limits:

* **Handwritten Notes / Code**: Uses `microsoft/trocr-base-handwritten` (Transformer-based OCR) running locally with PyTorch. Line-level segmentation + per-line recognition with beam search.
* **Printed Documents**: Uses `Tesseract OCR` for robust local character-by-character recognition of standard fonts.

## Main Technologies

* **Streamlit**: Web interface and interactive application framework
* **Hugging Face Transformers**: Model loading and inference (`VisionEncoderDecoderModel`)
* **PyTorch**: Local tensor operations and model execution
* **OpenCV**: Computer vision for deskewing, binarization, and horizontal projection profile (HPP) line segmentation
* **Tesseract / Pytesseract**: Local OCR for printed documents
* **FPDF2 & python-docx**: Document generation and export

## Local Installation

Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
git clone https://github.com/Vansh21jaiswal/handwritten-document-intelligence.git
cd handwritten-document-intelligence

# Install required dependencies
pip install -r requirements.txt

# (Optional) Install Tesseract on your system for Printed Document support
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr
```

## Local Run Command

To start the application locally:

```bash
# Run the Streamlit app
python3 -m streamlit run app/main.py
```

## Deployment

This app is designed to run on **Hugging Face Spaces** (free tier, 16 GB RAM). To deploy:

1. Create a new Space on [huggingface.co](https://huggingface.co/new-space) with SDK set to **Streamlit**.
2. Push this repository to the Space's git remote.
3. The app will auto-deploy — no API keys or secrets required.

## Known Limitations

* **Model Download**: On the first run, the TrOCR-Base model (~334 MB) will be downloaded from the Hugging Face Hub. Subsequent runs use the cached model.
* **Handwritten Cursive**: Extremely dense or highly stylized cursive handwriting will degrade accuracy.
* **Language Support**: Currently, the TrOCR model is optimized for English handwriting only.
* **Processing Time**: On CPU, each line takes ~2–3 seconds. A full page with 15 lines takes ~30–45 seconds.

## Screenshots

*(Screenshots placeholder)*

## GitHub Project Structure

```text
handwritten-document-intelligence/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── packages.txt              # System-level dependencies (Tesseract)
├── app/
│   ├── main.py               # Main Streamlit application entry point
├── scripts/
│   ├── run_handwriting_demo.py # Core ML pipeline and segmentation logic
├── notebooks/                # Jupyter notebooks for experimentation
└── src/                      # Additional source code utilities
```
