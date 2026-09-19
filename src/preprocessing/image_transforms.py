from __future__ import annotations
from typing import Dict, Any

import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from src.preprocessing.domain_transforms import DomainAdaptationTransforms

class GaussianNoise:
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std
            tensor = tensor + noise
        return tensor

class ImagePreprocessor:
    def __init__(
        self,
        target_height: int = 128,
        mean: float = 0.5,
        std: float = 0.5,
        augment: bool = False,
        aug_cfg: dict | None = None,
        domain_aug_cfg: dict | None = None,
    ) -> None:
        if target_height <= 0:
            raise ValueError(f"target_height must be positive, got {target_height}")
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")

        self.target_height = target_height
        self.mean = mean
        self.std = std
        self.augment = augment
        
        self.aug_transforms = None
        self.noise_transform = None
        self.domain_transform = None
        
        if self.augment:
            aug_cfg = aug_cfg or {}
            
            # Domain Adaptation Augmentations
            if domain_aug_cfg and domain_aug_cfg.get("enabled", False):
                self.domain_transform = DomainAdaptationTransforms(domain_aug_cfg)
                
            # Generic Augmentations
            if aug_cfg.get("enabled", False):
                rot = aug_cfg.get("rotation_degrees", 2)
                shear = aug_cfg.get("shear_degrees", 5)
                brightness = aug_cfg.get("brightness_jitter", 0.2)
                contrast = aug_cfg.get("contrast_jitter", 0.2)
                noise_std = aug_cfg.get("noise_std", 0.05)
                
                self.aug_transforms = transforms.Compose([
                    transforms.RandomApply([
                        transforms.RandomAffine(degrees=rot, shear=shear, fill=255)
                    ], p=0.7),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=brightness, contrast=contrast)
                    ], p=0.5),
                ])
                if noise_std > 0:
                    self.noise_transform = GaussianNoise(std=noise_std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self._process(image)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self._process(image)

    def _process(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        
        # Domain adaptation (expects RGB)
        if self.augment and self.domain_transform is not None:
            image = self.domain_transform(image)
            
        # Grayscale
        image = image.convert("L")

        if self.augment and self.aug_transforms is not None:
            image = self.aug_transforms(image)

        orig_w, orig_h = image.size
        scale = self.target_height / orig_h
        new_w = max(1, round(orig_w * scale))
        image = image.resize((new_w, self.target_height), Image.LANCZOS)

        tensor = TF.to_tensor(image)
        
        if self.augment and self.noise_transform is not None:
            tensor = self.noise_transform(tensor)

        tensor = TF.normalize(tensor, mean=[self.mean], std=[self.std])
        return tensor

    @classmethod
    def from_config(cls, config: dict, augment: bool = False) -> "ImagePreprocessor":
        aug_cfg = config.get("augmentation", {})
        domain_aug_cfg = config.get("domain_augmentation", {})
        
        # We allow augment flag to enable augmentation if config sets enabled=True
        return cls(
            target_height=config.get("target_height", 128),
            mean=config.get("mean", 0.5),
            std=config.get("std", 0.5),
            augment=augment,
            aug_cfg=aug_cfg,
            domain_aug_cfg=domain_aug_cfg
        )
