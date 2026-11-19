# Stage 1 a0 vs. literature profile-matching baselines

Every model below was re-trained here on the same rows and folds as a0; the published numbers are in the last table and are not comparable one-to-one (different features or protocol).

## columbia — protocol `lowo`, features `stage1` (8368 decisions, 7 categorical + 55 numeric fields)

| model | Part A AUC | fold AUC mean ± sd | AUC within | log-loss | Brier | acc | mutual AUC | ΔAUC vs a0 [95% CI] | TOST ±0.03 | Δ mutual [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 (Stage 1 SRRM) | 0.5723 | 0.5835 ± 0.0574 | 0.5819 | 0.6722 | 0.2398 | 0.580 | 0.5801 | — | — | — |
| LR | 0.5579 | 0.5842 ± 0.0639 | 0.5562 | 0.6792 | 0.2429 | 0.578 | 0.5638 | -0.014 [-0.034, +0.005] | inconclusive | -0.016 [-0.047, +0.014] |
| FM | 0.5357 | 0.5430 ± 0.0460 | 0.5219 | 0.7352 | 0.2642 | 0.552 | 0.5643 | -0.037 [-0.056, -0.016] | worse | -0.016 [-0.047, +0.015] |
| DeepFM | 0.5452 | 0.5523 ± 0.0588 | 0.5385 | 0.7881 | 0.2709 | 0.553 | 0.5565 | -0.027 [-0.046, -0.007] | worse | -0.024 [-0.055, +0.008] |
| xDeepFM | 0.5487 | 0.5635 ± 0.0511 | 0.5558 | 0.7884 | 0.2621 | 0.571 | 0.5453 | -0.024 [-0.044, -0.004] | worse | -0.035 [-0.069, +0.001] |
| DCN-V2 | 0.5485 | 0.5613 ± 0.0566 | 0.5434 | 0.7705 | 0.2614 | 0.569 | 0.5571 | -0.023 [-0.044, -0.003] | worse | -0.023 [-0.055, +0.009] |
| AutoInt | 0.5600 | 0.5838 ± 0.0668 | 0.5632 | 0.6917 | 0.2466 | 0.588 | 0.5635 | -0.012 [-0.033, +0.008] | equivalent | -0.017 [-0.047, +0.014] |
| IPNN | 0.5443 | 0.5587 ± 0.0543 | 0.5541 | 0.9070 | 0.2787 | 0.558 | 0.5648 | -0.028 [-0.048, -0.006] | worse | -0.015 [-0.049, +0.015] |
| biDeepFM | 0.5448 | 0.5510 ± 0.0584 | 0.5221 | 0.8042 | 0.2750 | 0.557 | 0.5492 | -0.028 [-0.048, -0.007] | worse | -0.030 [-0.064, +0.001] |
| XGBoost | 0.5446 | 0.5636 ± 0.0646 | 0.5532 | 0.6862 | 0.2458 | 0.564 | 0.5567 | -0.027 [-0.048, -0.006] | worse | -0.024 [-0.061, +0.014] |
| LightGBM | 0.5416 | 0.5547 ± 0.0659 | 0.5534 | 0.6861 | 0.2459 | 0.569 | 0.5419 | -0.030 [-0.051, -0.009] | worse | -0.038 [-0.074, -0.002] |
| CatBoost | 0.5357 | 0.5636 ± 0.0879 | 0.5432 | 0.6827 | 0.2447 | 0.568 | 0.5191 | -0.037 [-0.059, -0.015] | worse | -0.061 [-0.101, -0.021] |

## Published reference numbers (read from the papers, not re-measured)

| paper (venue, rank) | model | protocol | target | AUC | sd | mutual / match AUC |
|---|---|---|---|---|---|---|
| biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2 | biDeepFM_Base | 5-fold, participant-level | dec (Part A) | 0.6823 | 0.0234 | 0.7105 |
| biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2 | biLorentzFM_Base | 5-fold, participant-level | dec (Part A) | 0.7012 | 0.0189 | 0.7289 |
| LFKL, NeurIPS 2025 workshop (not ranked) | Logistic Regression (pre-date) | not stated | match | — | — | 0.6100 |
| LFKL, NeurIPS 2025 workshop (not ranked) | Similarity (cosine) | not stated | match | — | — | 0.5400 |
| McElfresh et al., NeurIPS 2023 D&B (A*) | XGBoost > CatBoost > LightGBM (rank) | OpenML 10 random folds, during-date features | match | — | — | — |
