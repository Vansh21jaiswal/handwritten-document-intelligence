# Handwritten Document Intelligence

Convert handwritten notebook pages, printed documents, and handwritten code into editable digital text.

## Overview

This application provides an AI-assisted end-to-end pipeline for recognizing text from images. It handles three primary types of documents:
1. **Handwritten Notes (Prose)**: English handwritten notebook pages.
2. **Handwritten Code / Math**: Complex handwritten programming syntax and math expressions.
3. **Printed / Digital Documents**: Clean digital text from certificates, ID cards, and printed pages.

*Note: Handwritten text recognition is inherently challenging. While the system uses state-of-the-art models, difficult cursive, ambiguous symbols, and messy handwriting may still require manual review. The application provides an interface to review and correct low-confidence lines.*

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

Depending on the selected mode, the application routes the image to different AI models:

* **Handwritten Notes**: Uses `microsoft/trocr-large-handwritten` (Transformer-based optical character recognition) running locally.
* **Printed Documents**: Uses `Tesseract OCR` for robust local character-by-character recognition of standard fonts.
* **Handwritten Code**: Uses **Cloud AI (Gemini 3.6 Flash)** to accurately parse complex, non-prose handwritten programming syntax (requires API key).

## Main Technologies

* **Streamlit**: Web interface and interactive application framework
* **Hugging Face Transformers**: Model loading and inference (`VisionEncoderDecoderModel`)
* **PyTorch**: Local tensor operations and model execution (supports Apple Silicon MPS)
* **OpenCV**: Computer vision for deskewing, binarization, and horizontal projection profile (HPP) line segmentation
* **Tesseract / Pytesseract**: Local OCR for printed documents
* **Google Generative AI SDK**: Cloud API connection for code recognition
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
# Optional: Set Gemini API key for Handwritten Code mode
export GEMINI_API_KEY="your_api_key_here"

# Run the Streamlit app
python3 -m streamlit run app/main.py
```

## Known Limitations

* **Model Download**: On the first run, the TrOCR-Large model (~3.3GB) will be downloaded from the Hugging Face Hub to your local cache.
* **Handwritten Cursive**: Extremely dense or highly stylized cursive handwriting will degrade accuracy.
* **Language Support**: Currently, the TrOCR model is optimized for English handwriting only.
* **Code Mode Requirement**: The "Handwritten Code" mode explicitly requires an active internet connection and a valid `GEMINI_API_KEY` exported in the environment.

## Screenshots

*(Screenshots placeholder)*

## GitHub Project Structure

```text
handwritten-document-intelligence/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── app/
│   ├── main.py               # Main Streamlit application entry point
├── scripts/
│   ├── run_handwriting_demo.py # Core ML pipeline and segmentation logic
├── notebooks/                # Jupyter notebooks for experimentation
└── src/                      # Additional source code utilities
```
