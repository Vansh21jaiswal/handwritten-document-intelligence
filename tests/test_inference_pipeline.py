import pytest
from pathlib import Path
from PIL import Image
import torch
import yaml

from src.inference.pipeline import HTRPipeline

# Use the existing baseline checkpoint for the smoke test
CHECKPOINT_PATH = Path("checkpoints/best.pt")
CONFIG_PATH = Path("configs/default.yaml")

@pytest.fixture
def dummy_image(tmp_path):
    """Creates a temporary dummy image for testing."""
    img = Image.new("RGB", (256, 128), color="white")
    path = tmp_path / "dummy.png"
    img.save(path)
    return path

@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Baseline checkpoint not found")
def test_pipeline_checkpoint_loading():
    """Verify that the pipeline successfully loads the checkpoint and reconstructs components."""
    pipeline = HTRPipeline(checkpoint_path=CHECKPOINT_PATH, config_path=CONFIG_PATH, device="cpu")
    
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None
    assert pipeline.preprocessor is not None
    
    # Check that the model is in eval mode
    assert not pipeline.model.training

@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Baseline checkpoint not found")
def test_pipeline_preprocessing_shape(dummy_image):
    """Verify the pipeline preprocessor outputs the correct shape."""
    pipeline = HTRPipeline(checkpoint_path=CHECKPOINT_PATH, config_path=CONFIG_PATH, device="cpu")
    
    img = Image.open(dummy_image)
    tensor = pipeline.preprocessor(img)
    
    # Preprocessor should output (1, H, W)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 1
    
    # Config specifies target_height
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    assert tensor.shape[1] == cfg["preprocessing"]["target_height"]

@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Baseline checkpoint not found")
def test_pipeline_deterministic_smoke_test(dummy_image):
    """Verify inference runs end-to-end and is deterministic."""
    pipeline = HTRPipeline(checkpoint_path=CHECKPOINT_PATH, config_path=CONFIG_PATH, device="cpu")
    
    # Inference 1
    pred1 = pipeline.predict(dummy_image)
    
    # Inference 2
    pred2 = pipeline.predict(dummy_image)
    
    # Output must be a string
    assert isinstance(pred1, str)
    
    # Deterministic check
    assert pred1 == pred2

@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Baseline checkpoint not found")
def test_pipeline_decoding_logic():
    """Verify the pipeline delegates to GreedyDecoder properly."""
    pipeline = HTRPipeline(checkpoint_path=CHECKPOINT_PATH, config_path=CONFIG_PATH, device="cpu")
    
    # Monkey patch decoder and model
    decode_called = False
    def mock_decode(logits):
        nonlocal decode_called
        decode_called = True
        return "mocked_prediction"
        
    pipeline.decoder.decode_single = mock_decode
    
    # Create random synthetic logits mimicking model output (T, 1, V)
    T, V = 10, pipeline.tokenizer.vocab_size
    mock_logits = torch.randn(T, 1, V)
    
    def mock_call(tensor):
        return mock_logits
        
    pipeline.model.__call__ = mock_call
    
    pred = pipeline.predict(Image.new("RGB", (100, 30)))
    assert decode_called
    assert pred == "mocked_prediction"

