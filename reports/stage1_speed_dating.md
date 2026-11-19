# Stage 1 — speed_dating

Rows 8360, dyads 4176, base rate 0.420, match rate 0.165. Cross-fit leave-one-wave-out; bootstrap by chooser; SESOI ΔAUC = 0.03 (TOST).

Locked SRRM: s = 1.497, r = -0.051, σa² = 0.710, σb² = 0.530, σab = -0.175, ρ = 0.235; R̂ = {'beta_max': 1.0026754190877825, 'sigma_max': 1.052066653113258, 'rho': 1.0118577819705732}; chosen = {'scale': 0.05, 'mask': 0.0, 'loss': 0.6734515197945712, 'se': 0.004392212507129829}

| branch | AUC | AUC within | Brier | ECE | log-loss | acc | match AUC | ΔAUC vs a0 [95% CI] | TOST ±0.03 |
|---|---|---|---|---|---|---|---|---|---|
| a0_SRRM | 0.564 | 0.565 | 0.2407 | 0.013 | 0.6742 | 0.579 | 0.562 | — | — |
