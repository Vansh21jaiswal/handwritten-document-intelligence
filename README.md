---
title: Intelligent Handwritten Document Recognition
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: app/main.py
pinned: false
---

# Intelligent Handwritten Document Recognition System

A production-quality AIML pipeline that extracts, evaluates, and structures text from both handwritten and printed documents. 

This project demonstrates a decoupled Microservice/Monorepo architecture, combining traditional Computer Vision (OpenCV), baseline OCR (Tesseract), and Deep Learning OCR (TrOCR) with automated image quality validation and structured JSON extraction.

## Features

- **Automated Image Quality Validation**: Detects extreme blur (Laplacian variance) and low contrast (Pixel STD) before passing images to heavy OCR models.
- **Model Comparison**: A/B test a traditional OCR baseline (Tesseract) against a Deep Learning model (TrOCR).
- **Line Segmentation Pipeline**: Employs OpenCV deskewing, binarization, and Horizontal Projection Profile (HPP) to isolate text lines.
- **Structured Data Extraction**: Automatically extracts entities (Dates, Amounts, Emails, Phones, Key-Values) from raw OCR text into structured JSON.
- **Microservice Architecture**: Cleanly decoupled `FastAPI` backend for predictions and a `Streamlit` frontend for visualization.
- **Zero API Dependencies**: All models run locally. No rate limits or cloud quotas.

## Architecture Pipeline

```text
Input Image
     ↓
Image Quality Validation (Blur / Contrast checks)
     ↓
Document Preprocessing (Deskewing / HPP Line Segmentation)
     ↓
Handwriting Recognition Engine (Tesseract Baseline OR TrOCR Deep Learning)
     ↓
Confidence Estimation 
     ↓
Structured Extraction (Regex / NLP post-processing)
     ↓
JSON Output (API) / Visual Dashboard (UI)
```

## Technologies Used

- **Deep Learning OCR**: `microsoft/trocr-base-handwritten` (HuggingFace / PyTorch)
- **Traditional OCR**: `Tesseract` (pytesseract)
- **Computer Vision**: OpenCV (Deskew, Binarization, Edge Detection)
- **Backend API**: FastAPI, Uvicorn
- **Frontend UI**: Streamlit
- **Testing**: Pytest
- **Deployment**: Docker, Hugging Face Spaces

## Evaluation & Model Comparison

*Note: The following metrics were collected on a standard CPU inference environment using the IAM Handwriting dataset test split.*

| Model | Architecture | Speed (CPU) | Accuracy (Handwriting) | Best Use Case |
|-------|--------------|-------------|------------------------|---------------|
| **Tesseract** | LSTM-based OCR | Fast (< 1s) | Poor | Clean, Printed Digital Text |
| **TrOCR Base** | VisionEncoderDecoder | Moderate (~2-3s/line) | High | Cursive, Notebooks, Messy Handwriting |

## Installation & Running Locally

Ensure you have Python 3.9+ installed. You also need Tesseract installed on your system.
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt install tesseract-ocr`

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Vansh21jaiswal/handwritten-document-intelligence.git
cd handwritten-document-intelligence
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard (UI)
```bash
python3 -m streamlit run app/main.py
```

### 3. Run the FastAPI Backend (API)
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
You can access the interactive API documentation at `http://localhost:8000/docs`.

## Docker Deployment

To build and run the backend using Docker:

```bash
docker build -t handwritten-recognition-api .
docker run -p 8000:8000 handwritten-recognition-api
```

## Limitations

- **Compute Heavy**: TrOCR runs slowly on CPUs without a dedicated GPU or MPS hardware acceleration. 
- **Cursive Variability**: Extreme medical handwriting or deeply cursive scripts will result in lower confidence scores.

## Future Improvements

- Add a dedicated NER model (Named Entity Recognition) to replace Regex in the Structured Extraction layer.
- Integrate GPU support explicitly into the Docker build using `nvidia-docker`.
