import pytest
import torch
from PIL import Image

from src.preprocessing.image_transforms import ImagePreprocessor


def test_augmentation_is_training_only():
    """Verify that augmentation is enabled only when specified."""
    cfg = {
        "target_height": 128,
        "augmentation": {
            "enabled": True,
            "rotation_degrees": 2,
            "shear_degrees": 5,
        }
    }
    
    # Train preprocessor (augment=True)
    train_prep = ImagePreprocessor.from_config(cfg, augment=True)
    assert train_prep.augment is True
    assert train_prep.aug_transforms is not None
    
    # Val/Test preprocessor (augment=False)
    val_prep = ImagePreprocessor.from_config(cfg, augment=False)
    assert val_prep.augment is False
    assert val_prep.aug_transforms is None


def test_validation_preprocessing_remains_deterministic():
    """Verify that validation preprocessing (augment=False) is deterministic."""
    img = Image.new("RGB", (300, 100), color="white")
    
    cfg = {"target_height": 128, "augmentation": {"enabled": True}}
    val_prep = ImagePreprocessor.from_config(cfg, augment=False)
    
    # Run twice, should be exactly the same
    tensor1 = val_prep(img)
    tensor2 = val_prep(img)
    
    assert torch.equal(tensor1, tensor2), "Validation preprocessing is not deterministic!"


def test_output_tensor_shape_remains_correct():
    """Verify augmented image outputs maintain correct height and channels."""
    img = Image.new("RGB", (300, 100), color="black")
    
    cfg = {
        "target_height": 128,
        "augmentation": {
            "enabled": True,
            "rotation_degrees": 15,
            "shear_degrees": 10,
            "noise_std": 0.1,
        }
    }
    train_prep = ImagePreprocessor.from_config(cfg, augment=True)
    
    tensor = train_prep(img)
    
    # Grayscale: 1 channel
    assert tensor.shape[0] == 1
    # Height fixed to target_height
    assert tensor.shape[1] == 128
    # Width should scale proportionally
    expected_w = max(1, round(300 * (128 / 100)))
    assert tensor.shape[2] == expected_w


def test_labels_are_unchanged_by_augmentation():
    """Labels are text, they don't change, but we ensure dataset returns same labels."""
    from src.dataset.tokenizer import CharTokenizer
    from src.dataset.iam_dataset import IAMTorchDataset
    from datasets import Dataset

    # Mock HF dataset row
    hf_ds = Dataset.from_dict({
        "image": [Image.new("RGB", (100, 30), color="white")],
        "text": ["hello"],
    })
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello"])
    
    cfg = {"target_height": 128, "augmentation": {"enabled": True, "noise_std": 0.5}}
    
    # Unaugmented
    val_prep = ImagePreprocessor.from_config(cfg, augment=False)
    ds_val = IAMTorchDataset(hf_ds, val_prep, tokenizer)
    
    # Augmented
    train_prep = ImagePreprocessor.from_config(cfg, augment=True)
    ds_train = IAMTorchDataset(hf_ds, train_prep, tokenizer)
    
    item_val = ds_val[0]
    item_train = ds_train[0]
    
    # Targets should be strictly identical
    assert torch.equal(item_val["target"], item_train["target"])
    assert item_val["text"] == item_train["text"]
    assert item_val["target_length"] == item_train["target_length"]
    
    # Tensors should differ due to aggressive noise
    assert not torch.allclose(item_val["image"], item_train["image"])
