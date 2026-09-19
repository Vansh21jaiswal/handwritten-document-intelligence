import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import matplotlib.pyplot as plt
from src.dataset.iam_loader import load_iam_splits
from src.preprocessing.image_transforms import ImagePreprocessor
import torchvision.transforms.functional as TF

def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    clean_prep = ImagePreprocessor.from_config(cfg["preprocessing"], augment=False)
    aug_prep = ImagePreprocessor.from_config(cfg["preprocessing"], augment=True)
    cache_dir = cfg.get("dataset", {}).get("cache_dir", None)
    splits = load_iam_splits(cache_dir=cache_dir)
    train_split = splits["train"]
    samples = [train_split[i] for i in range(5)]
    fig, axes = plt.subplots(5, 2, figsize=(16, 12))
    fig.suptitle("Data Augmentation Comparison (Training Set)", fontsize=16)
    axes[0, 0].set_title("Original (Val/Test deterministic)")
    axes[0, 1].set_title("Augmented (Training)")
    for i, sample in enumerate(samples):
        pil_img = sample["image"]
        text = sample["text"]
        t_clean = clean_prep(pil_img)
        t_aug = aug_prep(pil_img)
        def to_img(t):
            t = t * 0.5 + 0.5
            return TF.to_pil_image(t)
        img_clean = to_img(t_clean)
        img_aug = to_img(t_aug)
        axes[i, 0].imshow(img_clean, cmap="gray", aspect="auto")
        axes[i, 0].axis("off")
        axes[i, 0].text(0, -10, f"GT: {text}", fontsize=10, color="blue")
        axes[i, 1].imshow(img_aug, cmap="gray", aspect="auto")
        axes[i, 1].axis("off")
    plt.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "augmentation_example.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
