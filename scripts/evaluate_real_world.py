import sys
import glob
from pathlib import Path
import cv2
import numpy as np
import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.pipeline import HTRPipeline

def create_visualization(results, out_path):
    images = []
    labels = []
    max_w = 0
    total_h = 0
    margin = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    for file_path, pred_text in results:
        img = cv2.imread(str(file_path))
        if img is None: continue
        images.append(img)
        labels.append(pred_text)
        max_w = max(max_w, img.shape[1])
        (text_w, text_h), _ = cv2.getTextSize(pred_text, font, font_scale, font_thickness)
        max_w = max(max_w, text_w)
        total_h += img.shape[0] + text_h + 30 
    max_w += (margin * 2)
    total_h += margin
    canvas = np.ones((total_h, max_w, 3), dtype=np.uint8) * 255
    current_y = margin
    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        canvas[current_y:current_y+h, margin:margin+w] = img
        current_y += h + 25
        cv2.putText(canvas, label, (margin, current_y), font, font_scale, (0, 0, 0), font_thickness)
        current_y += 30 
    cv2.imwrite(out_path, canvas)

def main():
    pipeline = HTRPipeline(
        checkpoint_path="checkpoints/best.pt",
        config_path="configs/default.yaml",
        device="cpu"
    )
    iam_sample = Path("test_sample.png")
    if iam_sample.exists():
        pred = pipeline.predict(iam_sample)
        print("=== IAM Baseline Verification ===")
        print(f"Image: {iam_sample}")
        print(f"Prediction: {pred}\n")
    
    crop_files = sorted(glob.glob("outputs/segmentation/line_*.jpg"))
    # Filter out contact sheet
    crop_files = [cf for cf in crop_files if re.match(r'.*line_\d{3}\.jpg', cf)]
    if not crop_files: return
        
    results = []
    print("=== Real-World Notebook Inference ===")
    for cf in crop_files:
        pred = pipeline.predict(cf)
        results.append((cf, pred))
        print(f"{Path(cf).name}: {pred}")
        
    md_path = Path("outputs/segmentation/htr_predictions.md")
    with open(md_path, "w") as f:
        f.write("# HTR Predictions (Qualitative Domain-Shift Test)\n\n")
        f.write("| Filename | Predicted Transcription |\n")
        f.write("| :--- | :--- |\n")
        for cf, pred in results:
            safe_pred = pred.replace('|', '\\|')
            f.write(f"| {Path(cf).name} | `{safe_pred}` |\n")
    
    vis_path = "outputs/segmentation/htr_predictions.jpg"
    create_visualization(results, vis_path)

if __name__ == "__main__":
    main()
