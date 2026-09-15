# data/

This directory holds all project data. **None of these files are tracked by Git.**

## Layout

| Path | Description |
|------|-------------|
| `raw/` | Original, unmodified source images and annotations |
| `processed/` | Preprocessed images ready for model consumption |

## Notes

- Place raw handwritten document scans under `raw/`.
- Processed outputs (resized, binarized, normalized) go under `processed/`.
- Do not commit data files; see the root `.gitignore`.
