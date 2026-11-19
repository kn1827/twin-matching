# Diagnosis: why a0 is below the published Speed Dating baselines

## columbia

a0 (Stage 1 SRRM, LOWO) AUC = 0.5723. Latent R² of the profile part η = 0.024 (σa² = 0.73, σb² = 0.53). Two-way fixed effects (in-sample, knows every a_i and b_j): AUC = 0.896. Partner popularity alone under person5: AUC = 0.705. Persons with a unique profile vector: 98.9%.

| protocol | features | model | Part A AUC | AUC within | mutual AUC |
|---|---|---|---|---|---|
| lowo | stage1 | logit | 0.6072 | 0.5907 | 0.5915 |
| lowo | stage1 | lightgbm | 0.5545 | 0.5450 | 0.5583 |
| lowo | +attr | logit | 0.6114 | 0.5923 | 0.5923 |
| lowo | +attr | lightgbm | 0.5652 | 0.5491 | 0.5640 |
| lowo | literature | logit | 0.6257 | 0.6063 | 0.6018 |
| lowo | literature | lightgbm | 0.5940 | 0.5807 | 0.5824 |
| lowo | literature+partner_pop | logit | 0.6257 | 0.6065 | 0.6020 |
| lowo | literature+partner_pop | lightgbm | 0.5940 | 0.5807 | 0.5824 |
| person5 | stage1 | logit | 0.6305 | 0.6335 | 0.6149 |
| person5 | stage1 | lightgbm | 0.6623 | 0.7106 | 0.6486 |
| person5 | +attr | logit | 0.6372 | 0.6338 | 0.6166 |
| person5 | +attr | lightgbm | 0.6630 | 0.7035 | 0.6531 |
| person5 | literature | logit | 0.6521 | 0.6520 | 0.6252 |
| person5 | literature | lightgbm | 0.6819 | 0.7107 | 0.6716 |
| person5 | literature+partner_pop | logit | 0.7032 | 0.7489 | 0.6646 |
| person5 | literature+partner_pop | lightgbm | 0.7054 | 0.7515 | 0.6674 |

## speed_dating

a0 (Stage 1 SRRM, LOWO) AUC = 0.5642. Latent R² of the profile part η = 0.018 (σa² = 0.71, σb² = 0.53). Two-way fixed effects (in-sample, knows every a_i and b_j): AUC = 0.896. Partner popularity alone under person5: AUC = 0.704. Persons with a unique profile vector: 98.9%.

| protocol | features | model | Part A AUC | AUC within | mutual AUC |
|---|---|---|---|---|---|
| lowo | stage1 | logit | 0.5877 | 0.5793 | 0.5696 |
| lowo | stage1 | lightgbm | 0.4940 | 0.5116 | 0.4965 |
| lowo | +attr | logit | 0.5930 | 0.5796 | 0.5708 |
| lowo | +attr | lightgbm | 0.4929 | 0.5247 | 0.4866 |
| lowo | literature | logit | 0.6116 | 0.6004 | 0.5789 |
| lowo | literature | lightgbm | 0.5082 | 0.5187 | 0.5059 |
| lowo | literature+partner_pop | logit | 0.6116 | 0.6003 | 0.5790 |
| lowo | literature+partner_pop | lightgbm | 0.5082 | 0.5187 | 0.5059 |
| person5 | stage1 | logit | 0.6490 | 0.6640 | 0.6339 |
| person5 | stage1 | lightgbm | 0.6282 | 0.6764 | 0.6136 |
| person5 | +attr | logit | 0.6559 | 0.6678 | 0.6351 |
| person5 | +attr | lightgbm | 0.6324 | 0.6892 | 0.6141 |
| person5 | literature | logit | 0.6659 | 0.6787 | 0.6308 |
| person5 | literature | lightgbm | 0.6377 | 0.6805 | 0.6077 |
| person5 | literature+partner_pop | logit | 0.7008 | 0.7415 | 0.6562 |
| person5 | literature+partner_pop | lightgbm | 0.6890 | 0.7492 | 0.6413 |

