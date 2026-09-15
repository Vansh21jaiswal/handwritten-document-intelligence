# Handwritten Document Intelligence

A Python-based system for **Handwritten Text Recognition (HTR)** and document intelligence.

## Overview

This project aims to build an end-to-end pipeline for recognizing and interpreting handwritten text in document images. The initial MVP focuses on:

- **Image Preprocessing** — denoising, binarization, and normalization of handwritten document scans
- **Text Recognition** — deep-learning-based handwriting recognition
- **Evaluation** — Character Error Rate (CER) and Word Error Rate (WER) metrics
- **Inference** — a clean pipeline for running predictions on new images

>  This project is under active development. No accuracy claims are made at this stage.

## Project Structure

```
handwritten-document-intelligence/
├── data/               # Raw and processed data (not tracked by Git)
├── notebooks/          # Exploratory analysis and experiments
├── src/                # Core source code
│   ├── preprocessing/  # Image preprocessing utilities
│   ├── dataset/        # Dataset loading and handling
│   ├── models/         # Model architectures
│   ├── training/       # Training loops and utilities
│   ├── evaluation/     # CER / WER evaluation
│   └── inference/      # Inference pipeline
├── tests/              # Unit and integration tests
├── configs/            # Configuration files (YAML / JSON)
├── scripts/            # One-off utility scripts
└── app/                # Web interface (planned)
```

## Requirements

- Python 3.9+
- See [`requirements.txt`](requirements.txt) for dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT
