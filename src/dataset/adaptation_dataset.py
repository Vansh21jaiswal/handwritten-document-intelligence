import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.preprocessing.image_transforms import ImagePreprocessor
from src.dataset.tokenizer import CharTokenizer

class AdaptationDataset(Dataset):
    """
    PyTorch Dataset for real-world domain adaptation.
    Reads image/text pairs from a CSV file.
    """
    def __init__(
        self,
        csv_path: str,
        preprocessor: ImagePreprocessor,
        tokenizer: CharTokenizer,
    ) -> None:
        if not tokenizer.is_built():
            raise RuntimeError("Tokenizer vocabulary has not been built.")
            
        self.csv_path = Path(csv_path)
        self.img_dir = self.csv_path.parent / "images"
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        
        self.samples = []
        
        if self.csv_path.exists():
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_name = row.get("image_path", row.get("image", "")).strip()
                    text = row.get("text", "").strip()
                    
                    if not img_name or not text:
                        continue
                        
                    img_path = self.img_dir / img_name
                    if img_path.exists():
                        self.samples.append({"image": img_path, "text": text})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        text = sample["text"]
        
        # Load image
        pil_image = Image.open(sample["image"]).convert("RGB")
        
        # Image -> normalised tensor (1, H, W)
        image_tensor = self.preprocessor(pil_image)
        image_width = image_tensor.shape[2]
        
        # Text -> encoded integer tensor
        encoded = self.tokenizer.encode(text)
        target_tensor = torch.tensor(encoded, dtype=torch.long)
        
        return {
            "image": image_tensor,
            "target": target_tensor,
            "target_length": len(encoded),
            "text": text,
            "image_width": image_width,
        }
