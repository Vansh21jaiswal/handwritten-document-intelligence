import cv2
import numpy as np
from typing import List, Tuple

class DocumentSegmenter:
    """
    OpenCV-based handwritten document segmenter.
    Uses Hough deskewing and Horizontal Projection Profile (HPP) for robust line detection.
    """
    def __init__(self, target_width: int = 1000):
        self.target_width = target_width

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Lightweight deterministic deskew using Hough lines on horizontal structures."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
        
        angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only care about roughly horizontal lines
                if -25 < angle < 25:
                    angles.append(angle)
                    
        if len(angles) > 0:
            median_angle = np.median(angles)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image.copy()

    def segment_into_lines(self, image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Segments a document image into individual line crops.
        Returns the list of cropped images and a debug visualization image.
        """
        # 1. Resize for consistent processing
        h, w = image.shape[:2]
        scale = self.target_width / float(w)
        new_h = int(h * scale)
        resized = cv2.resize(image, (self.target_width, new_h))
        
        # 2. Deskew
        deskewed = self._deskew(resized)
        debug_vis = deskewed.copy()

        # 3. Grayscale and noise reduction
        gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # 4. Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        # 5. Conservative Ruled Line Removal
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.target_width // 10, 1))
        detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical_strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        binary_no_rules = cv2.subtract(binary, detect_horizontal)
        binary_cleaned = cv2.bitwise_or(binary_no_rules, vertical_strokes)

        # 6. Horizontal Projection Profile (HPP)
        hpp = np.sum(binary_cleaned, axis=1) / 255
        
        # Smooth HPP to prevent fragmentation
        smoothed_hpp = cv2.GaussianBlur(hpp.reshape(-1, 1), (15, 1), 0).flatten()
        
        # 7. Find text bands
        # Dynamic threshold based on active text density
        threshold = np.max(smoothed_hpp) * 0.05 if np.max(smoothed_hpp) > 0 else 1
        active_rows = smoothed_hpp > threshold
        
        bands = []
        y = 0
        from itertools import groupby
        for val, group in groupby(active_rows):
            length = sum(1 for _ in group)
            if val and length > 15: # minimum height for a line
                bands.append((y, y + length))
            y += length

        # 8. Split heavily merged bands (Touching lines)
        # Expected line height is roughly target_width // 15 (e.g., ~60px)
        expected_h = self.target_width // 15
        refined_bands = []
        for (y1, y2) in bands:
            if (y2 - y1) > expected_h * 2.5:
                # Find the deepest valley in the middle section of the HPP
                sub_hpp = smoothed_hpp[y1:y2]
                mid_start = int(len(sub_hpp) * 0.3)
                mid_end = int(len(sub_hpp) * 0.7)
                if mid_start < mid_end:
                    min_idx = np.argmin(sub_hpp[mid_start:mid_end]) + mid_start
                    cut_y = y1 + min_idx
                    refined_bands.append((y1, cut_y))
                    refined_bands.append((cut_y, y2))
                else:
                    refined_bands.append((y1, y2))
            else:
                refined_bands.append((y1, y2))

        # 9. Tighter horizontal and vertical bounds for each band
        crops = []
        for (y1, y2) in refined_bands:
            band_img = binary_cleaned[y1:y2, :]
            
            # Vertical tightness
            row_sums = np.sum(band_img, axis=1)
            nz_y = np.nonzero(row_sums)[0]
            if len(nz_y) == 0:
                continue
            ty1 = y1 + nz_y[0]
            ty2 = y1 + nz_y[-1]
            
            # Skip if too short after tightening
            if (ty2 - ty1) < 10:
                continue
                
            # Horizontal tightness
            band_img_tight = binary_cleaned[ty1:ty2, :]
            col_sums = np.sum(band_img_tight, axis=0)
            nz_x = np.nonzero(col_sums)[0]
            if len(nz_x) == 0:
                continue
            tx1 = max(0, nz_x[0] - 10)
            tx2 = min(deskewed.shape[1], nz_x[-1] + 10)
            
            # Final padded coordinates
            pad = 5
            fy1 = max(0, ty1 - pad)
            fy2 = min(deskewed.shape[0], ty2 + pad)
            
            crop = deskewed[fy1:fy2, tx1:tx2]
            crops.append(crop)
            
            # Draw on debug image
            cv2.rectangle(debug_vis, (tx1, fy1), (tx2, fy2), (0, 255, 0), 2)

        return crops, debug_vis

