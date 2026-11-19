# Stage 1 a0 vs. literature profile-matching baselines

Every model below was re-trained here on the same rows and folds as a0; the published numbers are in the last table and are not comparable one-to-one (different features or protocol).

## columbia — protocol `person5`, features `literature` (8368 decisions, 10 categorical + 73 numeric fields)

| model | Part A AUC | fold AUC mean ± sd | AUC within | log-loss | Brier | acc | mutual AUC | ΔAUC vs a0 [95% CI] | TOST ±0.03 | Δ mutual [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 (Stage 1 SRRM) | 0.5868 | 0.5897 ± 0.0136 | 0.5949 | 0.6692 | 0.2383 | 0.585 | 0.5995 | — | — | — |
| LR | 0.6494 | 0.6518 ± 0.0277 | 0.6629 | 0.6512 | 0.2299 | 0.608 | 0.6322 | +0.063 [+0.043, +0.081] | better | +0.033 [+0.001, +0.064] |
| FM | 0.6574 | 0.6576 ± 0.0142 | 0.6909 | 0.6965 | 0.2409 | 0.631 | 0.6399 | +0.071 [+0.051, +0.091] | better | +0.040 [+0.005, +0.075] |
| DeepFM | 0.6622 | 0.6751 ± 0.0178 | 0.7155 | 0.9994 | 0.2744 | 0.631 | 0.6282 | +0.076 [+0.054, +0.096] | better | +0.028 [-0.007, +0.064] |
| xDeepFM | 0.6555 | 0.6657 ± 0.0338 | 0.6973 | 0.7345 | 0.2453 | 0.618 | 0.6226 | +0.069 [+0.048, +0.088] | better | +0.023 [-0.010, +0.056] |
| DCN-V2 | 0.6668 | 0.6732 ± 0.0332 | 0.7065 | 1.0481 | 0.2722 | 0.628 | 0.6361 | +0.080 [+0.060, +0.099] | better | +0.036 [+0.003, +0.072] |
| AutoInt | 0.6450 | 0.6523 ± 0.0213 | 0.6685 | 0.6741 | 0.2363 | 0.615 | 0.6289 | +0.058 [+0.037, +0.078] | better | +0.030 [-0.005, +0.063] |
| IPNN | 0.6573 | 0.6707 ± 0.0171 | 0.7023 | 1.0063 | 0.2786 | 0.622 | 0.6278 | +0.071 [+0.049, +0.091] | better | +0.028 [-0.006, +0.062] |
| biDeepFM | 0.6522 | 0.6639 ± 0.0130 | 0.7081 | 0.8840 | 0.2625 | 0.624 | 0.6067 | +0.066 [+0.043, +0.088] | better | +0.008 [-0.026, +0.042] |
| XGBoost | 0.6769 | 0.6803 ± 0.0110 | 0.7006 | 0.6342 | 0.2218 | 0.633 | 0.6488 | +0.090 [+0.070, +0.110] | better | +0.049 [+0.019, +0.081] |
| LightGBM | 0.6940 | 0.6948 ± 0.0117 | 0.7202 | 0.6227 | 0.2169 | 0.647 | 0.6647 | +0.108 [+0.087, +0.127] | better | +0.065 [+0.034, +0.094] |
| CatBoost | 0.6848 | 0.6855 ± 0.0223 | 0.7060 | 0.6295 | 0.2198 | 0.641 | 0.6496 | +0.098 [+0.078, +0.116] | better | +0.050 [+0.021, +0.080] |

## Published reference numbers (read from the papers, not re-measured)

| paper (venue, rank) | model | protocol | target | AUC | sd | mutual / match AUC |
|---|---|---|---|---|---|---|
| biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2 | biDeepFM_Base | 5-fold, participant-level | dec (Part A) | 0.6823 | 0.0234 | 0.7105 |
| biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2 | biLorentzFM_Base | 5-fold, participant-level | dec (Part A) | 0.7012 | 0.0189 | 0.7289 |
| LFKL, NeurIPS 2025 workshop (not ranked) | Logistic Regression (pre-date) | not stated | match | — | — | 0.6100 |
| LFKL, NeurIPS 2025 workshop (not ranked) | Similarity (cosine) | not stated | match | — | — | 0.5400 |
| McElfresh et al., NeurIPS 2023 D&B (A*) | XGBoost > CatBoost > LightGBM (rank) | OpenML 10 random folds, during-date features | match | — | — | — |
