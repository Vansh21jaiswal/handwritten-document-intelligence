import sys
import glob
from pathlib import Path
import cv2
import numpy as np
import csv
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.document_segmentation import DocumentSegmenter

def main():
    selected_files = [
        "data/raw/WhatsApp Image 2026-09-19 at 13.05.52 (1).jpeg",
        "data/raw/WhatsApp Image 2026-09-19 at 13.05.53 (1).jpeg",
        "data/raw/WhatsApp Image 2026-09-19 at 13.05.53 (2).jpeg",
        "data/raw/WhatsApp Image 2026-09-19 at 13.05.53 (3).jpeg",
        "data/raw/WhatsApp Image 2026-09-19 at 13.05.54.jpeg",
    ]
    
    out_img_dir = Path("data/adaptation/images")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = Path("data/adaptation/labels.csv")
    
    segmenter = DocumentSegmenter(target_width=1000)
    
    global_idx = 1
    records = []
    
    print("Segmenting selected pages...")
    for fpath in selected_files:
        img = cv2.imread(fpath)
        if img is None:
            print(f"Warning: Could not read {fpath}")
            continue
            
        crops, _ = segmenter.segment_into_lines(img)
        source_name = Path(fpath).name
        
        for crop in crops:
            line_id = f"line_{global_idx:03d}.jpg"
            out_path = out_img_dir / line_id
            cv2.imwrite(str(out_path), crop)
            
            # Default text logic
            text = ""
            status = "NEEDS_REVIEW"
            
            if global_idx == 1:
                text = "Upcoming cycle counts"
                status = "VERIFIED"
            elif global_idx == 2:
                text = "missed / pending counts"
                status = "VERIFIED"
                
            records.append({
                "image_path": line_id,
                "text": text,
                "source_page": source_name,
                "status": status
            })
            global_idx += 1
            
    print(f"Generated {len(records)} line crops.")
    
    # Write CSV (only the 3 requested columns)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text", "source_page"])
        for r in records:
            writer.writerow([r["image_path"], r["text"], r["source_page"]])
            
    # Write Markdown
    out_md = Path("outputs/adaptation/label_review.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Adaptation Dataset Label Review\n\n")
        for r in records:
            f.write(f"### {r['image_path']}\n")
            f.write(f"- **Source:** `{r['source_page']}`\n")
            f.write(f"- **Status:** {r['status']}\n")
            # Wrap in codeblock to preserve spaces
            f.write(f"- **Transcription:**\n```\n{r['text']}\n```\n\n")
            
    # Create Contact Sheet
    print("Generating contact sheet...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    margin = 30
    
    # We will build a very tall canvas
    # Each row: [ Crop Image ] [ Text space ]
    max_w = 0
    total_h = margin
    
    row_data = []
    for r in records:
        img_path = out_img_dir / r["image_path"]
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        max_w = max(max_w, w)
        
        # Space for label box on the right (approx 400px)
        # We also put the line ID above the crop
        row_h = h + 40 # 30px for ID text, 10px padding
        total_h += row_h + margin
        row_data.append((img, r, row_h))
        
    canvas_w = max_w + 500 + margin*3 # 500 for writing space
    canvas = np.ones((total_h, canvas_w, 3), dtype=np.uint8) * 255
    
    current_y = margin
    for img, r, row_h in row_data:
        h, w = img.shape[:2]
        
        # Draw ID
        cv2.putText(canvas, r["image_path"], (margin, current_y + 20), font, font_scale, (0,0,0), thickness)
        
        # Paste Image
        canvas[current_y + 30 : current_y + 30 + h, margin : margin + w] = img
        
        # Draw Verification Box
        box_x = margin + max_w + margin
        box_w = 400
        box_h = h + 30
        cv2.rectangle(canvas, (box_x, current_y), (box_x + box_w, current_y + box_h), (200,200,200), 2)
        cv2.putText(canvas, "Verified Transcription:", (box_x + 10, current_y + 20), font, 0.5, (100,100,100), 1)
        
        # If verified, draw the text
        if r["status"] == "VERIFIED":
            cv2.putText(canvas, r["text"], (box_x + 10, current_y + 50), font, font_scale, (0,150,0), thickness)
        
        current_y += row_h + margin
        
    contact_path = "outputs/adaptation/label_review_contact_sheet.jpg"
    cv2.imwrite(contact_path, canvas)
    
    print(f"Saved {contact_path}")
    print(f"Saved {out_md}")
    print(f"Saved {out_csv}")
    
    print(f"\nSummary:")
    print(f"  Pages inspected: 8")
    print(f"  Pages selected: 5")
    print(f"  Line crops: {len(records)}")
    print(f"  Verified: {sum(1 for r in records if r['status'] == 'VERIFIED')}")
    print(f"  Needs Review: {sum(1 for r in records if r['status'] == 'NEEDS_REVIEW')}")

if __name__ == "__main__":
    main()
