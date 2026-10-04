import cv2
import numpy as np
from PIL import Image, ExifTags
from itertools import groupby

class ImageSegmenter:
    """
    Handles deskewing, normalization, and line segmentation for handwritten documents.
    Extracts individual line crops while filtering out background/noise.
    """

    def __init__(self, target_width: int = 1200, min_line_height: int = 15):
        self.target_width = target_width
        self.min_line_height = min_line_height

    def auto_rotate(self, image_path: str) -> np.ndarray:
        """
        Corrects image rotation using EXIF data.
        Returns a BGR numpy array ready for OpenCV.
        """
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

        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    def normalise_page(self, image: np.ndarray) -> np.ndarray:
        """Resizes to target width and applies lightweight deskew (Hough Transform)."""
        h, w = image.shape[:2]
        scale = self.target_width / float(w)
        new_h = int(h * scale)
        resized = cv2.resize(image, (self.target_width, new_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        
        edges = cv2.Canny(gray_eq, 30, 120, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                minLineLength=self.target_width // 15, maxLineGap=30)
        
        angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)

        if angles:
            median_angle = float(np.median(angles))
            if abs(median_angle) > 0.3:
                centre = (self.target_width // 2, new_h // 2)
                M = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
                resized = cv2.warpAffine(resized, M, (self.target_width, new_h),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
        return resized

    def detect_lines(self, page: np.ndarray) -> list:
        """Returns bounding boxes (y1, y2, x1, x2) for candidate text lines."""
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        blurred = cv2.bilateralFilter(gray_eq, 9, 75, 75)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10)

        # Remove horizontal ruled lines to prevent bridging text lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.target_width // 10, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        v_strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        
        cleaned = cv2.bitwise_or(cv2.subtract(binary, h_lines), v_strokes)

        # Horizontal Projection Profile (HPP)
        hpp = np.sum(cleaned, axis=1) / 255.0
        smoothed = cv2.GaussianBlur(hpp.reshape(-1, 1), (15, 1), 0).flatten()

        threshold = max(np.max(smoothed) * 0.05, 1.0)
        active = smoothed > threshold

        bands = []
        y = 0
        for val, grp in groupby(active):
            length = sum(1 for _ in grp)
            if val and length > self.min_line_height:
                bands.append((y, y + length))
            y += length

        expected_h = self.target_width // 15
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

        boxes.sort(key=lambda b: b[0]) # Top-to-bottom
        return boxes

    def categorize_crop(self, crop_bgr: np.ndarray) -> tuple:
        """Determines if a crop contains handwriting, or should be rejected."""
        h, w = crop_bgr.shape[:2]
        area = h * w
        if h < 12 or area < 400:
            return "rejected", "too small/short"

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        dark_ratio = np.sum(gray < 60) / area

        if mean_sat > 40 and dark_ratio > 0.3:
            return "rejected", f"desk edge (sat={mean_sat:.0f})"
        if dark_ratio > 0.6:
            return "rejected", f"too dark ({dark_ratio:.2f})"

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray_eq = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 10)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 4, 30), 1))
        h_only = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        strokes = cv2.subtract(binary, h_only)

        ink_density = np.sum(strokes > 0) / area

        if ink_density < 0.001:
            return "rejected", f"blank (ink={ink_density:.4f})"

        uncertain_reasons = []
        if ink_density < 0.008:
            uncertain_reasons.append(f"low ink ({ink_density:.3f})")

        aspect = w / h if h > 0 else 0
        if aspect > 30:
            uncertain_reasons.append(f"very wide (aspect={aspect:.1f})")

        contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meaningful = [c for c in contours if cv2.contourArea(c) > 8]
        if len(meaningful) < 2:
            uncertain_reasons.append(f"few contours ({len(meaningful)})")

        if uncertain_reasons:
            return "uncertain", " | ".join(uncertain_reasons)

        return "accepted", "good"
