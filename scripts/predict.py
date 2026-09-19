#!/usr/bin/env python3
"""
predict.py
==========
Command-line inference script for the Handwritten Text Recognition model.

Usage
-----
    python3 scripts/predict.py --image path/to/image.png
    python3 scripts/predict.py --image path/to/image.png --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import HTRPipeline


def main():
    parser = argparse.ArgumentParser(description="Run HTR inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the handwritten line image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (e.g., cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}", file=sys.stderr)
        sys.exit(1)
        
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading checkpoint : {ckpt_path}")
    print(f"Loading image      : {img_path}")
    
    # Initialize pipeline
    pipeline = HTRPipeline(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Run prediction
    prediction = pipeline.predict(img_path)
    
    print("-" * 50)
    print("Prediction:")
    print(f"  {prediction}")
    print("-" * 50)
    
    # Note: Current CTC greedy decoding doesn't produce a calibrated sequence-level confidence score trivially. 
    # To invent one would require product of frame probabilities which is often dominated by blank tokens.
    # Therefore, following requirement #9, we omit a confidence score.


if __name__ == "__main__":
    main()
