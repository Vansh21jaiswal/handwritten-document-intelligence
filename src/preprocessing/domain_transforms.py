import cv2
import numpy as np
import random
from PIL import Image

class DomainAdaptationTransforms:
    """
    Applies realistic domain-adaptive augmentations to simulate
    photographed notebook handwriting variations.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prob = cfg.get("probability", 0.5)

    def _add_ruled_lines(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.cfg.get("ruled_line_prob", 0.5):
            return img
            
        h, w = img.shape[:2]
        out = img.copy()
        
        # Determine number of lines
        num_lines = random.randint(1, 3)
        y_positions = sorted([random.randint(0, h-1) for _ in range(num_lines)])
        
        for y in y_positions:
            color = random.randint(150, 220)  # light gray/blue line
            thickness = random.randint(1, 2)
            cv2.line(out, (0, y), (w, y), (color, color, color), thickness)
            
        return out

    def _uneven_illumination(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.cfg.get("illumination_prob", 0.5):
            return img
            
        h, w = img.shape[:2]
        # Create a gradient shadow mask
        gradient = np.zeros((h, w), dtype=np.float32)
        
        # Shadow origin
        shadow_type = random.choice(["linear_x", "linear_y", "radial"])
        if shadow_type == "linear_x":
            start = random.uniform(0.5, 1.0)
            end = random.uniform(0.1, 0.5)
            gradient = np.linspace(start, end, w, dtype=np.float32)
            gradient = np.tile(gradient, (h, 1))
        elif shadow_type == "linear_y":
            start = random.uniform(0.5, 1.0)
            end = random.uniform(0.1, 0.5)
            gradient = np.linspace(start, end, h, dtype=np.float32)
            gradient = np.tile(gradient, (w, 1)).T
        else:
            # simple center radial
            cx, cy = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            gradient = 1.0 - 0.5 * (dist / max_dist)
            
        out = (img.astype(np.float32) * np.expand_dims(gradient, axis=2))
        return np.clip(out, 0, 255).astype(np.uint8)
        
    def _blur_and_jpeg(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.cfg.get("blur_prob", 0.3):
            return img
        
        # Random blur
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
        
        # Random JPEG compression
        quality = random.randint(30, 80)
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.imdecode(enc, 1)
        return img
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
            
        # Convert PIL to OpenCV (RGB to BGR)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Apply sequential transforms
        cv_img = self._add_ruled_lines(cv_img)
        cv_img = self._uneven_illumination(cv_img)
        cv_img = self._blur_and_jpeg(cv_img)
        
        # Convert back
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
