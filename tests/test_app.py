import pytest
from pathlib import Path

# Mock streamlit before importing app
import sys
from unittest.mock import MagicMock

mock_st = MagicMock()
# Make cache_resource a pass-through decorator
def mock_cache_resource(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

mock_st.cache_resource = mock_cache_resource
sys.modules['streamlit'] = mock_st

from app.app import load_model_pipeline
from src.inference.pipeline import HTRPipeline

def test_load_model_pipeline():
    """Verify that the app successfully loads the HTRPipeline instance."""
    pipeline = load_model_pipeline()
    assert isinstance(pipeline, HTRPipeline)
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None
