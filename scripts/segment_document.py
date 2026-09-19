#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import cv2
import os

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.document_segmentation import DocumentSegmenter

def main():
    parser = argparse.ArgumentParser(description="Segment a handwritten document into individual lines.")
    parser.add_argument("--image", type=str, required=True, help="Path to the document image")
    parser.add_argument("--outdir", type=str, default="outputs/segmentation", help="Directory to save crops and debug visualization")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image {img_path}")
        sys.exit(1)

    print(f"Processing image: {img_path} (shape: {img.shape})")

    # Segment
    segmenter = DocumentSegmenter(target_width=1000)
    
    # We can access internal deskew for visualization
    h, w = img.shape[:2]
    scale = segmenter.target_width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(img, (segmenter.target_width, new_h))
    deskewed = segmenter._deskew(resized)

    crops, debug_vis = segmenter.segment_into_lines(img)

    print(f"Detected {len(crops)} lines.")

    # Save original (resized)
    cv2.imwrite(str(outdir / "01_original.jpg"), resized)
    
    # Save deskewed
    cv2.imwrite(str(outdir / "02_deskewed.jpg"), deskewed)

    # Save bounding boxes
    cv2.imwrite(str(outdir / "03_detected_bboxes.jpg"), debug_vis)
    
    print(f"Saved debug visualizations (original, deskewed, bboxes) to {outdir}")

    # Save crops
    for i, crop in enumerate(crops):
        crop_path = outdir / f"line_{i:03d}.jpg"
        cv2.imwrite(str(crop_path), crop)

    print(f"Saved {len(crops)} line crops to {outdir}")

if __name__ == "__main__":
    main()
