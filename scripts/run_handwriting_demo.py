#!/usr/bin/env python3
"""
End-to-end Handwritten Document Intelligence Demo (v2).

Improvements over v1:
  - Crop quality filter rejects non-handwriting regions BEFORE recognition
  - Debug image shows accepted (green) and rejected (red) lines
  - Lines ordered strictly top-to-bottom

Pipeline:
  notebook page photograph
       ↓
  page normalisation (resize, mild deskew)
       ↓
  line detection (HPP on binary mask)
       ↓
  crop quality filter (reject blanks, borders, ruled-line-only, headers)
       ↓
  individual line crops (raw colour — no binarisation)
       ↓
  pretrained TrOCR recognition (zero-shot)
       ↓
  ordered text output (JSON + plain text + debug visualisation)

Usage:
  python scripts/run_handwriting_demo.py --image data/raw/<image>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from itertools import groupby


# ── 0. Auto-rotation ─────────────────────────────────────────────────────────

def _hpp_score(candidate: np.ndarray) -> int:
    """Higher score = more distinct horizontal text lines."""
    gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    _, bw = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = bw.sum(axis=1).astype(np.float32)
    if proj.max() == 0:
        return 0
    proj /= proj.max()
    peaks = int(((proj > 0.15)[:-1] & ~(proj > 0.15)[1:]).sum())
    return peaks


def _osd_rotation(pil_img) -> tuple:
    """Use Tesseract OSD to detect content orientation (0, 90, 180, 270).
    Returns (angle_to_rotate, confidence).
    angle is degrees to rotate to make text upright (0 means already upright).
    Returns (0, 0.0) if OSD fails."""
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(pil_img, config="--psm 0 -c min_characters_to_try=5")
        angle = 0
        confidence = 0.0
        for line in osd.split("\n"):
            if "Rotate:" in line:
                angle = int(line.split(":")[1].strip())
            if "Orientation confidence:" in line:
                confidence = float(line.split(":")[1].strip())
        return angle, confidence
    except Exception:
        pass
    return 0, 0.0



def auto_rotate_image(image_path: str) -> np.ndarray:
    """Load image and correct its rotation using a multi-stage strategy.

    Stage 1: Apply EXIF orientation tag (phones set this without rotating pixels).
    Stage 2: Use Tesseract OSD to detect content orientation at 0/90/180/270°.
    Stage 3: If still landscape (w > h), pick the 90° rotation giving the most
             horizontal HPP peaks (the better portrait orientation).
    """
    # Stage 1: Apply EXIF orientation
    pil_img = Image.open(image_path)
    try:
        exif = pil_img._getexif()
        if exif:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 2:
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    pil_img = pil_img.rotate(180, expand=True)
                elif orientation == 4:
                    pil_img = pil_img.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    pil_img = pil_img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    pil_img = pil_img.rotate(-90, expand=True)
                elif orientation == 7:
                    pil_img = pil_img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    pil_img = pil_img.rotate(90, expand=True)
    except Exception:
        pass

    # Convert to OpenCV BGR
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Stage 2: Tesseract OSD — detect if content is rotated 90/180/270°
    osd_angle, osd_conf = _osd_rotation(pil_img)
    if osd_angle != 0:
        rotation_map = {
            90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_CLOCKWISE,
        }
        if osd_angle in rotation_map:
            img = cv2.rotate(img, rotation_map[osd_angle])
            print(f"OSD: rotate {osd_angle}° (confidence={osd_conf:.2f}) — corrected.")


    # Stage 3: If still landscape after EXIF + OSD, pick best 90° rotation by HPP score
    h, w = img.shape[:2]
    if w > h:
        cw  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        score_cw  = _hpp_score(cw)
        score_ccw = _hpp_score(ccw)
        img = cw if score_cw >= score_ccw else ccw
        print(f"Landscape fallback: rotated {'CW' if score_cw >= score_ccw else 'CCW'} "
              f"(CW score={score_cw}, CCW score={score_ccw}).")

    return img




def parse_args():
    p = argparse.ArgumentParser(
        description="Handwritten Document Intelligence — End-to-End Demo")
    p.add_argument("--image", required=True, help="Path to a notebook page photograph")
    p.add_argument("--output-dir", default="outputs/demo",
                   help="Directory for all outputs")
    p.add_argument("--target-width", type=int, default=1200,
                   help="Resize page width for consistent segmentation")
    p.add_argument("--min-line-height", type=int, default=15,
                   help="Minimum pixel height to count as a text line")
    return p.parse_args()


# ── 1. Page normalisation ─────────────────────────────────────────────────

def normalise_page(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize to a consistent width and apply lightweight deskew.

    No binarisation, no ruled-line removal, no CLAHE on the actual image.
    Deskew detection uses a CLAHE-enhanced grayscale for better edge detection
    on low-contrast photos (dark rooms, uneven lighting).
    Angle range extended to ±45° to handle notebook photos taken at a slant.
    """
    h, w = image.shape[:2]
    scale = target_width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_CUBIC)

    # Use CLAHE-enhanced grayscale ONLY for angle detection — raw pixels untouched
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    edges = cv2.Canny(gray_eq, 30, 120, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=target_width // 15, maxLineGap=30)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Accept near-horizontal lines (what we want to align with)
            if -45 < angle < 45:
                angles.append(angle)

    if angles:
        median_angle = float(np.median(angles))
        if abs(median_angle) > 0.3:
            centre = (target_width // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
            resized = cv2.warpAffine(resized, M, (target_width, new_h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            print(f"Deskewed by {median_angle:.1f}°")
    return resized




# ── 2. Line detection / segmentation ──────────────────────────────────────

def detect_lines(page: np.ndarray,
                 target_width: int,
                 min_line_height: int) -> list:
    """Return bounding boxes (y1, y2, x1, x2) for every candidate text line.

    CLAHE is applied to the grayscale copy for better ink detection on
    unevenly-lit photos (shadows, dark rooms, oblique flash).
    The original colour image is NOT modified.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    # CLAHE on greyscale ONLY — improves contrast for thresholding in dark/shadow areas
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    blurred = cv2.bilateralFilter(gray_eq, 9, 75, 75)


    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10)

    # Remove ruled lines from binary mask only (NOT from the colour image)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (target_width // 10, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    v_strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    cleaned = cv2.bitwise_or(cv2.subtract(binary, h_lines), v_strokes)

    hpp = np.sum(cleaned, axis=1) / 255.0
    smoothed = cv2.GaussianBlur(hpp.reshape(-1, 1), (15, 1), 0).flatten()

    threshold = max(np.max(smoothed) * 0.05, 1.0)
    active = smoothed > threshold

    bands = []
    y = 0
    for val, grp in groupby(active):
        length = sum(1 for _ in grp)
        if val and length > min_line_height:
            bands.append((y, y + length))
        y += length

    expected_h = target_width // 15
    refined = []
    for y1, y2 in bands:
        if (y2 - y1) > expected_h * 2.5:
            sub = smoothed[y1:y2]
            m_s, m_e = int(len(sub) * 0.3), int(len(sub) * 0.7)
            if m_s < m_e:
                cut = y1 + np.argmin(sub[m_s:m_e]) + m_s
                refined.append((y1, cut))
                refined.append((cut, y2))
            else:
                refined.append((y1, y2))
        else:
            refined.append((y1, y2))

    boxes = []
    for y1, y2 in refined:
        band = cleaned[y1:y2, :]
        row_sums = np.sum(band, axis=1)
        nz_y = np.nonzero(row_sums)[0]
        if len(nz_y) == 0:
            continue
        ty1 = y1 + nz_y[0]
        ty2 = y1 + nz_y[-1]
        if (ty2 - ty1) < 10:
            continue

        col_sums = np.sum(cleaned[ty1:ty2, :], axis=0)
        nz_x = np.nonzero(col_sums)[0]
        if len(nz_x) == 0:
            continue
        tx1 = max(0, nz_x[0] - 10)
        tx2 = min(page.shape[1], nz_x[-1] + 10)

        pad = 5
        fy1 = max(0, ty1 - pad)
        fy2 = min(page.shape[0], ty2 + pad)

        boxes.append((fy1, fy2, tx1, tx2))

    # Sort top-to-bottom by y1
    boxes.sort(key=lambda b: b[0])
    return boxes


# ── 3. Crop quality filter ────────────────────────────────────────────────

def categorize_crop(crop_bgr: np.ndarray) -> tuple:
    """Decide whether a crop contains handwriting (or code/symbols).

    Returns (status: str, reason: str).
    status can be 'accepted', 'uncertain', or 'rejected'.

    The filter is intentionally permissive — it only rejects obvious
    non-content (blank rows, desk edges, pure ruled lines).
    Code lines, math expressions, and sparse handwriting are passed as
    'uncertain' rather than rejected.
    """
    h, w = crop_bgr.shape[:2]
    area = h * w
    if h < 12 or area < 400:
        return "rejected", "too small/short"

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    dark_ratio = np.sum(gray < 60) / area

    # 1. Obvious desk/background (high saturation + very dark)
    if mean_sat > 40 and dark_ratio > 0.3:
        return "rejected", f"desk edge (sat={mean_sat:.0f})"
    if dark_ratio > 0.6:
        return "rejected", f"too dark ({dark_ratio:.2f})"

    # Use CLAHE-enhanced grayscale for better ink detection
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    # Remove long horizontal lines for ink calculation
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 30), 1))
    h_only = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    strokes = cv2.subtract(binary, h_only)

    ink_density = np.sum(strokes > 0) / area

    # 2. Completely blank rows — very low threshold to keep code symbols
    if ink_density < 0.001:
        return "rejected", f"blank (ink={ink_density:.4f})"

    # Mark uncertain (not rejected) for low-ink lines — could be sparse code
    uncertain_reasons = []

    if ink_density < 0.008:
        uncertain_reasons.append(f"low ink ({ink_density:.3f})")

    # Wide aspect is fine for code — don't penalise it
    aspect = w / h if h > 0 else 0
    if aspect > 30:
        uncertain_reasons.append(f"very wide (aspect={aspect:.1f})")

    contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Lower minimum contour area to catch small symbols like '.', ',', '{', '}'
    meaningful = [c for c in contours if cv2.contourArea(c) > 8]
    if len(meaningful) < 2:
        uncertain_reasons.append(f"few contours ({len(meaningful)})")

    if uncertain_reasons:
        return "uncertain", " | ".join(uncertain_reasons)

    return "accepted", "good"




# ── 4. TrOCR recognition ─────────────────────────────────────────────────

def load_recogniser(device):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_name = "microsoft/trocr-large-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def recognise_line(pil_image: Image.Image, processor, model, device) -> tuple:
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = model.generate(
            pixel_values, 
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            return_dict_in_generate=True,
            output_scores=True
        )
    ids = outputs.sequences
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    
    conf_score = 1.0
    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
        conf_score = torch.exp(outputs.sequences_scores[0]).item()
        
    return text, conf_score


# ── 5. Debug visualisation ────────────────────────────────────────────────
def draw_debug_image(page, boxes, statuses, output_path):
    """Draw accepted (green), uncertain (yellow), and rejected (red) bounding boxes."""
    vis = page.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    processed_num = 0
    for i, (y1, y2, x1, x2) in enumerate(boxes):
        status, reason = statuses[i]
        if status == "accepted":
            processed_num += 1
            colour = (0, 255, 0)
            label = f"L{processed_num}"
        elif status == "uncertain":
            processed_num += 1
            colour = (0, 255, 255)
            label = f"L{processed_num} (?)"
        else:
            colour = (0, 0, 255)
            label = f"X ({reason})"

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        # Use black text for yellow boxes for readability, white for others
        text_col = (0, 0, 0) if status == "uncertain" else (255, 255, 255)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), font, 0.45, text_col, 1)

    cv2.imwrite(output_path, vis)


# ── 6. Main pipeline ─────────────────────────────────────────────────────

def check_review_required(text: str, conf_score: float, crop_w: int) -> tuple:
    """Heuristics to flag suspicious recognitions for manual review."""
    text = text.strip()
    if not text:
        return True, "Empty output"
    
    if conf_score < 0.3:
        return True, "Very low generation confidence"
        
    if len(text) < 3 and crop_w > 100:
        return True, "Output unusually short"
        
    words = text.split()
    if len(words) > 4:
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.4:
            return True, "Suspicious repetition"
            
    alnum = sum(c.isalnum() for c in text)
    if len(text) > 0 and alnum < len(text) * 0.3:
        return True, "Mostly symbols/punctuation"
        
    return False, ""


def run_pipeline(image_path: str, out_dir: str, target_width: int = 1200, min_line_height: int = 15):
    """Run the complete end-to-end handwriting recognition pipeline."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, "line_crops")
    # Clean previous crops
    if os.path.exists(crops_dir):
        for f in os.listdir(crops_dir):
            if f.endswith(".jpg"):
                os.remove(os.path.join(crops_dir, f))
    os.makedirs(crops_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device:  {device}")
    print(f"Image:   {image_path}")
    print()

    # ── Load page (with automatic rotation correction) ─────────────────
    raw = auto_rotate_image(image_path)
    if raw is None:
        raise ValueError(f"Could not read image: {image_path}")
    print(f"Original size: {raw.shape[1]}×{raw.shape[0]}")


    # ── Step 1: Page normalisation ─────────────────────────────────────
    page = normalise_page(raw, target_width)
    print(f"Normalised:    {page.shape[1]}×{page.shape[0]}")

    # ── Step 2: Line detection ─────────────────────────────────────────
    boxes = detect_lines(page, target_width, min_line_height)
    print(f"Raw segments detected: {len(boxes)}")
    print()

    # ── Step 3: Crop quality filter ────────────────────────────────────
    statuses = []   # (status, reason) for every box
    processed = []  # (index, box, status) for accepted/uncertain boxes
    rejected = []   # (index, box, reason) for rejected boxes

    for i, (y1, y2, x1, x2) in enumerate(boxes):
        crop_bgr = page[y1:y2, x1:x2]
        status, reason = categorize_crop(crop_bgr)
        statuses.append((status, reason))
        
        if status in ("accepted", "uncertain"):
            processed.append((i, (y1, y2, x1, x2), status))
        else:
            rejected.append((i, (y1, y2, x1, x2), reason))

    accepted_count = sum(1 for s, _ in statuses if s == "accepted")
    uncertain_count = sum(1 for s, _ in statuses if s == "uncertain")

    print(f"Rejected by filter: {len(rejected)}")
    for idx, box, reason in rejected:
        print(f"  Segment {idx+1:2d}: {reason}")

    print(f"\nAccepted handwriting lines: {accepted_count}")
    print(f"Uncertain candidate lines: {uncertain_count}")
    print(f"Total lines sent to TrOCR: {len(processed)}")
    print()

    # ── Step 4: Load recogniser ────────────────────────────────────────
    print("Loading TrOCR (microsoft/trocr-large-handwritten) …")
    processor, model = load_recogniser(device)
    print("Model loaded.\n")

    # ── Step 5: Recognise each accepted line ───────────────────────────
    results = []
    t0 = time.time()
    review_count = 0

    for line_num, (orig_idx, (y1, y2, x1, x2), status) in enumerate(processed, 1):
        crop_bgr = page[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        crop_path = os.path.join(crops_dir, f"line_{line_num:03d}.jpg")
        cv2.imwrite(crop_path, crop_bgr)

        pred, conf = recognise_line(pil_crop, processor, model, device)
        
        needs_review, reason = check_review_required(pred, conf, x2 - x1)
        if needs_review or status == "uncertain":
            needs_review = True
            if not reason:
                reason = "Uncertain crop quality"
            review_count += 1

        results.append({
            "line_number": line_num,
            "segment_index": orig_idx + 1,
            "bbox": [int(y1), int(y2), int(x1), int(x2)],
            "crop_path": crop_path,
            "text": pred,
            "status": status,
            "review_required": needs_review,
            "review_reason": reason
        })
        flag = " [NEEDS REVIEW]" if needs_review else ""
        print(f"  Line {line_num:2d} ({status}): {pred}{flag}")

    elapsed = time.time() - t0
    print(f"\nRecognition completed in {elapsed:.1f}s")

    # ── Step 6: Combined transcription ─────────────────────────────────
    full_text = "\n".join(r["text"] for r in results)

    # ── Step 7: Save outputs ───────────────────────────────────────────
    output_json = {
        "model": "microsoft/trocr-large-handwritten",
        "image": str(image_path),
        "preprocessing": "resize + mild Hough deskew",
        "segmentation": "HPP on binary mask + crop quality filter",
        "device": str(device),
        "num_detected_lines": len(boxes),
        "num_accepted_lines": len(processed),
        "num_review_lines": review_count,
        "inference_time_seconds": round(elapsed, 1),
        "lines": results,
        "final_transcription": full_text
    }
    json_path = os.path.join(out_dir, "final_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(out_dir, "final_transcription.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Debug visualisation with accepted/rejected
    debug_path = os.path.join(out_dir, "final_segmentation.jpg")
    draw_debug_image(page, boxes, statuses, debug_path)
    
    return {
        "json_path": json_path,
        "txt_path": txt_path,
        "debug_path": debug_path,
        "results": output_json
    }

def main():
    args = parse_args()
    res = run_pipeline(args.image, args.output_dir, args.target_width, args.min_line_height)
    
    print(f"\n{'='*60}")
    print("OUTPUTS SAVED")
    print(f"  JSON result:       {res['json_path']}")
    print(f"  Plain text:        {res['txt_path']}")
    print(f"  Debug image:       {res['debug_path']}")
    print(f"{'='*60}")
    print(f"\n{'─'*60}")
    print("FULL TRANSCRIPTION")
    print(f"{'─'*60}")
    print(res['results']['final_transcription'])
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
