# Domain Adaptation Dataset

This folder holds the dataset for adapting the IAM baseline model to real-world notebook handwriting.

## Structure
- `images/`: Directory containing individual line crops (e.g., `line_001.jpg`).
- `labels.csv`: A CSV file defining the transcriptions.

## labels.csv Format
The CSV must have a header `image,text` and use commas.

```csv
image,text
line_001.jpg,Upcoming cycle counts
line_002.jpg,missed / pending counts
```

**Note:** Do not commit actual dataset images or labels to git.
