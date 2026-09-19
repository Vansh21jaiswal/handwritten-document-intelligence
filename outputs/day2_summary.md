# Day 2 Summary: Baseline & Augmentation Experiment

## Overview
During Day 2, we successfully trained, evaluated, and analyzed the baseline CNN-BiLSTM-CTC model, followed by a controlled data augmentation experiment.

### Architecture
*   **Backbone:** 4-block CNN (reduces image width by a factor of 4)
*   **Sequence Model:** 2-layer BiLSTM (256 hidden units)
*   **Decoder:** Linear projection to CTC (81 character vocabulary)
*   **Parameters:** ~3.05 Million

## 1. Baseline Performance
The reference model was trained on the complete IAM training split (6,482 samples) and evaluated on the 2,915-sample held-out IAM test split. 

*   **Test CER:** 12.12%
*   **Test WER:** 42.06%
*   **Evaluation Time:** ~2.5 minutes (CPU)

**Error Analysis Findings:**
*   The model achieves ~13.45% CER on medium-length sequences, but degrades severely on very short (<30) or very long (>60) sequences (21.52% CER).
*   Substitutions dominate the errors (~63%), indicating the model tracks sequence length well but fails to disambiguate visually overlapping/similar cursive characters (e.g., 'm' vs 'n').
*   Punctuation marks (like quotes, dashes, or question marks) frequently cause catastrophic decoding hallucinations.

## 2. Augmentation Experiment
Based on the error analysis (predominant substitutions and visual ambiguity), we implemented a training-time data augmentation pipeline to improve robustness. 

*   **Configuration:** Rotation (±2°), Shear (±5°), Brightness Jitter (±20%), Contrast Jitter (±20%), and Gaussian Noise injection (std 0.05).
*   **Constraints:** Augmentations were exclusively applied to training data. Test evaluation (2,915 samples) remained strictly deterministic. 

**Augmented Results:**
*   **Test CER:** 14.78% (worsened by +2.66% absolute / +21.9% relative)
*   **Test WER:** 43.47% (worsened by +1.41% absolute / +3.35% relative)

## Conclusion
**The controlled augmentation experiment did not improve generalization.**
Adding spatial and photometric noise degraded the small-capacity baseline (3M parameters). The architecture lacked the sufficient parameter count to learn a robust invariant representation, causing the noise to act as destructive interference against clean transcription targets. 

Because augmentation demonstrably degraded test performance, **the Baseline checkpoint (`checkpoints/best.pt`) has been retained as the official reference model** for the project, and data augmentation has been disabled in the default configuration.
