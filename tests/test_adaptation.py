import pytest
import os
import torch
import csv
from pathlib import Path
from PIL import Image
import numpy as np

from src.dataset.tokenizer import CharTokenizer
from src.preprocessing.image_transforms import ImagePreprocessor
from src.dataset.adaptation_dataset import AdaptationDataset
from src.inference.pipeline import HTRPipeline

@pytest.fixture
def dummy_adaptation_data(tmp_path):
    csv_path = tmp_path / "labels.csv"
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    img1_path = img_dir / "line_001.jpg"
    img2_path = img_dir / "line_002.jpg"
    
    img1 = Image.new('RGB', (100, 32), color='white')
    img1.save(img1_path)
    img2 = Image.new('RGB', (150, 32), color='white')
    img2.save(img2_path)
    
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text"])
        writer.writerow(["line_001.jpg", "hello"])
        writer.writerow(["line_002.jpg", "world"])
        writer.writerow(["line_003.jpg", "missing image"])
        writer.writerow(["line_004.jpg", ""])
        writer.writerow(["", "empty filename"])
        
    return csv_path

def test_adaptation_csv_parsing(dummy_adaptation_data):
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello", "world"])
    prep = ImagePreprocessor(target_height=32)
    dataset = AdaptationDataset(str(dummy_adaptation_data), prep, tokenizer)
    assert len(dataset) == 2

def test_adaptation_dataset_loading(dummy_adaptation_data):
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello", "world"])
    prep = ImagePreprocessor(target_height=32)
    dataset = AdaptationDataset(str(dummy_adaptation_data), prep, tokenizer)
    item = dataset[0]
    assert "image" in item
    assert item["image"].shape == (1, 32, 100)

def test_domain_augmentation_config():
    config = {
        "target_height": 32,
        "domain_augmentation": {
            "enabled": True,
            "probability": 1.0,
        }
    }
    prep_train = ImagePreprocessor.from_config(config, augment=True)
    assert prep_train.domain_transform is not None
    
    prep_val = ImagePreprocessor.from_config(config, augment=False)
    assert prep_val.domain_transform is None

def test_existing_inference_intact(tmp_path):
    import cv2, numpy as np
    test_img = str(tmp_path / "test_sample.png")
    cv2.imwrite(test_img, np.ones((32,100,3), dtype=np.uint8)*255)

    pipeline = HTRPipeline(checkpoint_path="checkpoints/best.pt", device="cpu")
    pred = pipeline.predict(test_img)
    assert isinstance(pred, str)
