# data/

This directory holds all project data. **None of these files are tracked by Git.**

## Layout

| Path | Description |
|------|-------------|
| `raw/` | Original, unmodified source images and annotations (if downloaded locally) |
| `processed/` | Preprocessed images ready for model consumption |

## Primary dataset: Teklia/IAM-line

The project uses the **IAM Handwriting Database** (line-level) for all HTR
training, validation, and evaluation.

| Property | Detail |
|---|---|
| **HuggingFace ID** | `Teklia/IAM-line` |
| **HF URL** | https://huggingface.co/datasets/Teklia/IAM-line |
| **Original source** | IAM Handwriting Database — University of Bern / HEIA-FR |
| **Paper** | Marti & Bunke, *IJDAR*, 2002 |
| **License** | Non-commercial research only |
| **Annotation level** | **Line-level** — one image + one transcription string per sample |

### Splits

| Split | Approx. samples | Purpose |
|---|---|---|
| `train` | ~6,482 | Model training |
| `validation` | ~976 | Hyper-parameter tuning |
| `test` | ~2,915 | Final CER / WER evaluation |

### Storage policy

**The dataset is NOT stored in this repository.**

- The HuggingFace `datasets` library downloads and caches files automatically
  under `~/.cache/huggingface/datasets/` on first use.
- Do not place dataset images or transcriptions under `data/raw/` or `data/processed/`.
- Both directories are listed in `.gitignore` to prevent accidental commits.

### Loading the dataset

```python
from src.dataset import load_iam_splits

splits = load_iam_splits()          # downloads on first call; cached thereafter
train_ds  = splits["train"]
val_ds    = splits["validation"]
test_ds   = splits["test"]

sample = train_ds[0]
print(sample["text"])               # ground-truth transcription string
image = sample["image"]             # PIL.Image (grayscale)
```

### Running the inspection script

```bash
python scripts/inspect_dataset.py --samples 5
```

### Running the visualization script

```bash
python scripts/visualize_samples.py --split train --n 8
# Headless: save to file instead of opening a window
python scripts/visualize_samples.py --split train --n 8 --save outputs/samples.png
```
