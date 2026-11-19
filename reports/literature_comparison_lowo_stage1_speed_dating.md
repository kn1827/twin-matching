# Stage 1 a0 vs. literature profile-matching baselines

Every model below was re-trained here on the same rows and folds as a0; the published numbers are in the last table and are not comparable one-to-one (different features or protocol).

## speed_dating — protocol `lowo`, features `stage1` (8352 decisions, 4 categorical + 55 numeric fields)

| model | Part A AUC | fold AUC mean ± sd | AUC within | log-loss | Brier | acc | mutual AUC | ΔAUC vs a0 [95% CI] | TOST ±0.03 | Δ mutual [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 (Stage 1 SRRM) | 0.5642 | 0.5852 ± 0.0776 | 0.5642 | 0.6743 | 0.2407 | 0.578 | 0.5617 | — | — | — |
| LR | 0.5551 | 0.5649 ± 0.0595 | 0.5569 | 0.6772 | 0.2420 | 0.586 | 0.5612 | -0.009 [-0.030, +0.012] | equivalent | -0.001 [-0.032, +0.030] |
| FM | 0.5396 | 0.5498 ± 0.0615 | 0.5344 | 0.7363 | 0.2625 | 0.569 | 0.5355 | -0.025 [-0.046, -0.003] | worse | -0.027 [-0.061, +0.006] |
| DeepFM | 0.5437 | 0.5420 ± 0.0542 | 0.5416 | 0.8330 | 0.2753 | 0.557 | 0.5492 | -0.020 [-0.041, -0.000] | worse | -0.013 [-0.046, +0.019] |
| xDeepFM | 0.5491 | 0.5604 ± 0.0541 | 0.5519 | 0.7827 | 0.2590 | 0.574 | 0.5386 | -0.015 [-0.036, +0.005] | inconclusive | -0.023 [-0.056, +0.009] |
| DCN-V2 | 0.5495 | 0.5581 ± 0.0510 | 0.5445 | 0.7430 | 0.2572 | 0.567 | 0.5431 | -0.015 [-0.037, +0.007] | inconclusive | -0.019 [-0.052, +0.015] |
| AutoInt | 0.5416 | 0.5656 ± 0.0623 | 0.5549 | 0.7032 | 0.2501 | 0.576 | 0.5347 | -0.022 [-0.045, -0.000] | worse | -0.027 [-0.061, +0.009] |
| IPNN | 0.5353 | 0.5503 ± 0.0518 | 0.5373 | 0.8543 | 0.2783 | 0.542 | 0.5466 | -0.029 [-0.053, -0.007] | worse | -0.015 [-0.045, +0.018] |
| biDeepFM | 0.5452 | 0.5616 ± 0.0591 | 0.5442 | 0.7416 | 0.2594 | 0.568 | 0.5498 | -0.019 [-0.043, +0.004] | inconclusive | -0.012 [-0.043, +0.021] |
| XGBoost | 0.4847 | 0.4854 ± 0.0735 | 0.4923 | 0.6866 | 0.2464 | 0.576 | 0.4682 | -0.079 [-0.107, -0.052] | worse | -0.093 [-0.134, -0.051] |
| LightGBM | 0.5118 | 0.5169 ± 0.0634 | 0.5313 | 0.6865 | 0.2463 | 0.575 | 0.5157 | -0.052 [-0.079, -0.024] | worse | -0.046 [-0.086, -0.009] |
| CatBoost | 0.5020 | 0.5154 ± 0.0344 | 0.5167 | 0.6929 | 0.2494 | 0.556 | 0.4817 | -0.062 [-0.090, -0.036] | worse | -0.079 [-0.120, -0.039] |

## Published reference numbers (read from the papers, not re-measured)

| paper (venue, rank) | model | protocol | target | AUC | sd | mutual / match AUC |
|---|---|---|---|---|---|---|
| biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2 | biDeepFM_Base | 5-fold, participant-level | dec (Part A) | 0.6823 | 0.0234 | 0.7105 |
| biLorentzFM, Applied Sciences 15(22):12340 (2025), MDPI, SJR Q2 | biLorentzFM_Base | 5-fold, participant-level | dec (Part A) | 0.7012 | 0.0189 | 0.7289 |
| LFKL, NeurIPS 2025 workshop (not ranked) | Logistic Regression (pre-date) | not stated | match | — | — | 0.6100 |
| LFKL, NeurIPS 2025 workshop (not ranked) | Similarity (cosine) | not stated | match | — | — | 0.5400 |
| McElfresh et al., NeurIPS 2023 D&B (A*) | XGBoost > CatBoost > LightGBM (rank) | OpenML 10 random folds, during-date features | match | — | — | — |
