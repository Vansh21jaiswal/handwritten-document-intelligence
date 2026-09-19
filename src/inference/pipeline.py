from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, Union

import torch
from PIL import Image

from src.models.crnn import CRNN
from src.dataset.tokenizer import CharTokenizer
from src.training.checkpoint import load_checkpoint, rebuild_tokenizer_from_checkpoint
from src.inference.greedy_decoder import GreedyDecoder
from src.preprocessing.image_transforms import ImagePreprocessor

class HTRPipeline:
    """End-to-end Handwritten Text Recognition inference pipeline.

    Loads a trained model checkpoint and its associated vocabulary,
    processes raw images, and returns predicted transcriptions.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path] = "configs/default.yaml",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        
        # 1. Load checkpoint (contains model config, weights, and tokenizer vocab)
        state = load_checkpoint(checkpoint_path, device=self.device)
        
        # 2. Rebuild tokenizer
        self.tokenizer = rebuild_tokenizer_from_checkpoint(state)
        
        # 3. Build model and load weights
        self.model = CRNN.from_config(self.tokenizer.vocab_size, state["model_cfg"])
        # We can just load the state directly since we already have it in 'state'
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Decoder
        self.decoder = GreedyDecoder(self.tokenizer)
        
        # 5. Preprocessor
        # Load preprocessing settings from config file to ensure consistency
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        # We explicitly set augment=False for deterministic inference
        self.preprocessor = ImagePreprocessor.from_config(cfg["preprocessing"], augment=False)


    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Run HTR inference on a single image.

        Parameters
        ----------
        image : str, Path, or PIL.Image
            The handwritten line image to transcribe.

        Returns
        -------
        str
            The predicted text.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            
        # Preprocess: PIL -> (1, H, W) tensor
        tensor = self.preprocessor(image)
        
        # Add batch dimension -> (1, 1, H, W)
        batch_tensor = tensor.unsqueeze(0).to(self.device)
        
        # Forward pass -> (T, 1, vocab_size)
        logits = self.model(batch_tensor)
        
        # Decode -> str
        prediction = self.decoder.decode_single(logits[:, 0, :])
        return prediction

