import cv2
import numpy as np

class ImageValidator:
    """
    Validates image quality before passing it to the OCR pipeline.
    Detects blur, poor contrast, and invalid resolutions.
    """
    
    def __init__(self, min_resolution=(200, 200), blur_threshold=50.0, contrast_threshold=20.0):
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold

    def validate(self, image_bgr: np.ndarray) -> dict:
        """
        Validates the input image.
        Returns a dictionary with 'is_valid', 'reason', and quality 'metrics'.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {"is_valid": False, "reason": "Empty or invalid image data.", "metrics": {}}

        h, w = image_bgr.shape[:2]
        if w < self.min_resolution[0] or h < self.min_resolution[1]:
            return {
                "is_valid": False,
                "reason": f"Image resolution ({w}x{h}) is too low for reliable OCR.",
                "metrics": {"width": w, "height": h}
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Blur Detection (Variance of Laplacian)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Contrast Detection (Standard Deviation of pixels)
        contrast_score = gray.std()

        metrics = {
            "blur_score": round(float(blur_score), 2),
            "contrast_score": round(float(contrast_score), 2),
            "width": w,
            "height": h
        }

        if blur_score < self.blur_threshold:
            return {
                "is_valid": False,
                "reason": f"Image is too blurry for reliable recognition (score: {metrics['blur_score']}).",
                "metrics": metrics
            }

        if contrast_score < self.contrast_threshold:
            return {
                "is_valid": False,
                "reason": f"Image contrast is too low (score: {metrics['contrast_score']}).",
                "metrics": metrics
            }

        return {"is_valid": True, "reason": "Image quality is acceptable.", "metrics": metrics}
