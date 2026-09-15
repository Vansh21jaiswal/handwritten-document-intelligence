"""
Dataset sub-package.

Handles dataset loading, tokenization, PyTorch Dataset/DataLoader
construction for the IAM Handwriting Database.

Public API
----------
load_iam_splits()       -> DatasetDict  (train / validation / test HF splits)
CharTokenizer           -> character-level vocab, encode, decode
IAMTorchDataset         -> torch.utils.data.Dataset
build_collate_fn()      -> collate fn for variable-width image batches

Quickstart
----------
>>> from src.dataset import load_iam_splits, CharTokenizer
>>> from src.dataset import IAMTorchDataset, build_collate_fn
>>> from src.preprocessing import ImagePreprocessor
>>> from torch.utils.data import DataLoader

>>> splits    = load_iam_splits()
>>> tokenizer = CharTokenizer()
>>> tokenizer.build_vocab(splits["train"]["text"])
>>> preprocessor = ImagePreprocessor(target_height=128)
>>> train_ds = IAMTorchDataset(splits["train"], preprocessor, tokenizer)
>>> loader   = DataLoader(train_ds, batch_size=16,
...                       collate_fn=build_collate_fn(), shuffle=True)
>>> batch = next(iter(loader))
"""

from src.dataset.iam_loader import load_iam_splits
from src.dataset.tokenizer import CharTokenizer
from src.dataset.iam_dataset import IAMTorchDataset, build_collate_fn

__all__ = [
    "load_iam_splits",
    "CharTokenizer",
    "IAMTorchDataset",
    "build_collate_fn",
]
