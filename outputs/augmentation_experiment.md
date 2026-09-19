# Day 2: Data Augmentation Experiment Report

## 1. Experiment Overview
**Goal:** Evaluate whether introducing realistic image augmentations (mild rotation, shear, brightness, contrast, and noise) during training improves generalization and reduces the baseline Test CER.

**Setup:**
*   **Architecture:** Identical CNN-BiLSTM-CTC (CRNN)
*   **Vocabulary:** 81 characters
*   **Training Data:** 6,482 samples
*   **Validation Data:** 976 samples
*   **Test Data:** 2,915 samples
*   **Hyperparameters:** 15 Epochs, batch size 8, lr 5e-4, AdamW, weight_decay 1e-4, gradient clip 5.0
*   **Augmentation Configuration:**
    *   Rotation: ±2°
    *   Shear: ±5°
    *   Brightness Jitter: ±20%
    *   Contrast Jitter: ±20%
    *   Gaussian Noise Std: 0.05
    *   *Applied exclusively to the training set; validation/test remained 100% deterministic.*

## 2. Checkpoint Selection
*   **Best Checkpoint:** `checkpoints/augmented/best.pt`
*   **Selected at Epoch:** 14
*   **Validation Loss:** 0.3472
*   **Total Training Time:** ~962.4 min (including background idle time)

## 3. Comparative Test Results
Evaluation performed on the held-out test split of 2,915 samples (Execution time: ~3.5 minutes).

| Metric | Baseline | Augmented | Absolute Change | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **Test CER** | 12.12% | **14.78%** | +2.66% | +21.95% |
| **Test WER** | 42.06% | **43.47%** | +1.41% | +3.35% |

### 🚨 Conclusion
**Data augmentation WORSENED the baseline.** 
The introduction of geometric distortions (affine/shear) and photometric distortions (jitter/noise) caused the test CER to regress from 12.12% up to 14.78%. 

**Analysis:** The CNN-BiLSTM-CTC baseline has a relatively small capacity (~3 million parameters). While augmentations typically improve generalization in large models, adding noise to a small capacity network can destroy the clear structural signal it needs to fit the data. The acoustic (visual) model became less confident in its features when faced with synthetic variations. A significantly higher capacity CNN backbone (like a ResNet) or longer training schedules would likely be required to benefit from this augmentation.

## 4. Sample Test Predictions (Augmented Model)

**[1]**
**GT:** `assuredness " Bella Bella Marie " ( Parlophone ) , a lively song that changes tempo mid-way .`
**PR:** `assuredness "bella Bella Harie " ( Parlophone ) , a lively song that cange , tempo mid-way .`
**CER:** 0.065 | **WER:** 0.278

**[2]**
**GT:** `I don't think he will storm the charts with this one , but it 's a good start .`
**PR:** `I don't thine he will storm the charts with this one , bat it 's a good slart .`
**CER:** 0.038 | **WER:** 0.158

**[3]**
**GT:** `CHRIS CHARLES , 39 , who lives in Stockton-on-Tees , is an accountant .`
**PR:** `CHlR's cinRLeS , 39 , who lives in stocuton-on- Tees,isan accounlant .`
**CER:** 0.197 | **WER:** 0.500

**[4]**
**GT:** `Become a success with a disc and hey presto ! You 're a star ... . Rolly sings with`
**PR:** `Become a success with a dise and hey presto ! Youire a slar.. . Rolly sings withh`
**CER:** 0.084 | **WER:** 0.316

**[5]**
**GT:** `Tolch , as he is known in Tin Pan Alley , likes songs with a month in the title . He wrote`
**PR:** `tolch , as he is unown in Fin Pan Alley , lines songs with a month in the tille . He wrotel`
**CER:** 0.067 | **WER:** 0.273

**[6]**
**GT:** `" My September Love , " the big David Whitfield hit of 1956 .`
**PR:** `"My Septemioer love " , the big Dovid whitpield hitor 19se .`
**CER:** 0.213 | **WER:** 0.714

**[7]**
**GT:** `He is also a director of a couple of garages . And he finds time as well to be a lyric`
**PR:** `He is also a director of a couple of garages . And he rinds time as well to be a lyric`
**CER:** 0.012 | **WER:** 0.048

**[8]**
**GT:** `writer . He writes with Tolchard Evans , composer of " Lady of Spain " and other big hits .`
**PR:** `writer . He writes with Tolchard Evans , compose , of "lady or goainl and other big hits .`
**CER:** 0.099 | **WER:** 0.350

**[9]**
**GT:** `The numbers include " Scotland the Brave , " " Men of Harlech , "`
**PR:** `The yumbers inelude " Seotland the brave, s alllen of Harde ch , s`
**CER:** 0.200 | **WER:** 0.733

**[10]**
**GT:** `Fay Compton stars in " No Hiding Place " (I T V , 9.35 p.m. ) .`
**PR:** `Fay Comgton stars in " No Hiding glace " (7 1 , 233p ni ..`
**CER:** 0.238 | **WER:** 0.529
