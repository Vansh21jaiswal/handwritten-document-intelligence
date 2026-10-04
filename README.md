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

A production-quality AIML pipeline that extracts, evaluates, and structures text from handwritten, printed, and code documents.

## Overview
This project converts physical document images into usable, structured digital data. It is engineered with a decoupled Microservice/Monorepo architecture, combining traditional Computer Vision (OpenCV), baseline OCR (Tesseract), and Cloud AI (Gemini Vision) with automated image quality validation and structured JSON extraction.

## Problem Statement
Converting handwritten notes, printed documents, and handwritten source code into editable digital text is challenging due to handwriting variability, messy backgrounds, and uneven lighting. Most OCR solutions fail on cursive or complex mathematical/programming syntax. This system solves that by integrating CV preprocessing with state-of-the-art AI.

## Features
- **Handwritten Recognition**: Extracts English handwriting with high fidelity.
- **Printed Text Recognition**: Baseline OCR for digital forms and certificates.
- **Code Recognition**: Accurately extracts handwritten programming syntax, preserving formatting.
- **Automatic Document Type Detection**: Heuristically detects Source Code, Forms, or General Text.
- **Image Quality Analysis**: Detects extreme blur (Laplacian variance) and low contrast (Pixel STD) before recognition.
- **Preprocessing Pipeline**: OpenCV deskewing, binarization, and Horizontal Projection Profile (HPP) line segmentation.
- **Structured Extraction**: Extracts Dates, Amounts, Emails, Phones, and Key-Values into JSON via Regex NLP processing.
- **Export System**: Export results to TXT, editable DOCX, professional PDF, and machine-readable JSON.
- **Model Comparison**: Swap between Traditional OCR (Tesseract) and AI Recognition.

## Architecture Pipeline

```text
Input Image
     ↓
Image Quality Analysis (Blur / Contrast checks)
     ↓
Computer Vision Preprocessing (Deskewing / HPP Segmentation)
     ↓
AI Recognition Engine (Tesseract or Gemini Vision)
     ↓
Text Post-Processing
     ↓
Structured Extraction (Regex / NLP extraction)
     ↓
Results UI / Export (PDF, DOCX, JSON)
```

## Technology Stack
- **Python** (Core Logic)
- **Streamlit** (Frontend Dashboard)
- **FastAPI / Uvicorn** (Backend Microservice structure)
- **OpenCV** (Computer Vision Preprocessing)
- **Tesseract (pytesseract)** (Traditional OCR)
- **Gemini Vision (google-generativeai)** (Cloud AI Recognition)
- **FPDF2 / python-docx** (Report Generation)
- **Pytest** (Testing)
- **Docker** (Containerization)

## Export Functionality
- **TXT**: Clean raw text output.
- **PDF**: Professional report containing metadata, confidence, structured information, and recognized text.
- **DOCX**: Fully editable Word document preserving layout where applicable.
- **JSON**: Machine-readable format for API integrations and database ingestion.

## Installation & Running Locally

Ensure you have Python 3.9+ installed, along with system-level Tesseract (`brew install tesseract` or `sudo apt install tesseract-ocr`).

```bash
# 1. Clone the repository
git clone https://github.com/Vansh21jaiswal/handwritten-document-intelligence.git
cd handwritten-document-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add Environment Variables (Create a secrets.toml or .env)
# GEMINI_API_KEY="your_api_key_here"

# 4. Run the Streamlit Application
python3 -m streamlit run app/main.py
```

## Deployment
The application is containerized and ready for Hugging Face Spaces or Streamlit Community Cloud.

**Live Demo:** [Streamlit Cloud Deployment](https://handwritten-document-intelligence-g6tcm8bjqvyvpe7osnxzmv.streamlit.app)

To run the backend API via Docker:
```bash
docker build -t intelligent-ocr .
docker run -p 8000:8000 intelligent-ocr
```

## Limitations
- **Image Quality Dependence**: Heavily blurred or overexposed images will significantly degrade recognition.
- **Handwriting Ambiguity**: Extremely messy cursive may result in hallucinated characters.
- **Confidence Metric**: Confidence scores provided by Cloud APIs are estimated and should not be interpreted as guaranteed mathematical accuracy (unlike localized token-probability metrics).
- **Processing Time**: Cloud AI inference depends on internet latency and API quotas (e.g., 5-20 seconds).

## Future Improvements
- Integrate a local Named Entity Recognition (NER) model (e.g., SpaCy) to replace Regex for more robust Structured Extraction.
- Re-integrate localized PyTorch models (TrOCR) if deployed to GPU-accelerated cloud infrastructure.
