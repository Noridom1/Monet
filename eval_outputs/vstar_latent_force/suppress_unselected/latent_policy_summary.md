# VStarBench controlled latent-activation evaluation

| X forced | Forced | Suppressed | Realized activation | Accuracy | Delta vs baseline | Force compliance | Suppression compliance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0/191 | 0/191 | 64/191 (33.51%) | 71.73% | +0.00 pp | n/a | n/a |
| 10% | 19/191 | 172/191 | 19/191 (9.95%) | 67.54% | -4.19 pp | 100.00% | 100.00% |
| 20% | 38/191 | 153/191 | 38/191 (19.90%) | 67.02% | -4.71 pp | 100.00% | 100.00% |
| 40% | 76/191 | 115/191 | 76/191 (39.79%) | 69.63% | -2.09 pp | 100.00% | 100.00% |
| 50% | 96/191 | 95/191 | 96/191 (50.26%) | 71.20% | -0.52 pp | 100.00% | 100.00% |
| 60% | 115/191 | 76/191 | 115/191 (60.21%) | 70.16% | -1.57 pp | 100.00% | 100.00% |
| 70% | 134/191 | 57/191 | 134/191 (70.16%) | 70.68% | -1.05 pp | 100.00% | 100.00% |
| 80% | 153/191 | 38/191 | 153/191 (80.10%) | 69.63% | -2.09 pp | 100.00% | 100.00% |
| 90% | 172/191 | 19/191 | 172/191 (90.05%) | 69.11% | -2.62 pp | 100.00% | 100.00% |
| 100% | 191/191 | 0/191 | 191/191 (100.00%) | 70.16% | -1.57 pp | 100.00% | n/a |

## Subset effects

| X | Subset | N | Accuracy | Paired delta vs baseline | Corrected | Broken |
|---:|---|---:|---:|---:|---:|---:|
| 10% | forced | 19 | 89.47% | +5.26 pp | 1 | 0 |
| 10% | forced_baseline_active | 11 | 90.91% | +9.09 pp | 1 | 0 |
| 10% | forced_baseline_inactive | 8 | 87.50% | +0.00 pp | 0 | 0 |
| 10% | suppressed | 172 | 65.12% | -5.23 pp | 1 | 10 |
| 10% | suppressed_baseline_active | 53 | 54.72% | -16.98 pp | 1 | 10 |
| 10% | suppressed_baseline_inactive | 119 | 69.75% | +0.00 pp | 0 | 0 |
| 20% | forced | 38 | 76.32% | -2.63 pp | 1 | 2 |
| 20% | forced_baseline_active | 18 | 72.22% | +5.56 pp | 1 | 0 |
| 20% | forced_baseline_inactive | 20 | 80.00% | -10.00 pp | 0 | 2 |
| 20% | suppressed | 153 | 64.71% | -5.23 pp | 1 | 9 |
| 20% | suppressed_baseline_active | 46 | 58.70% | -17.39 pp | 1 | 9 |
| 20% | suppressed_baseline_inactive | 107 | 67.29% | +0.00 pp | 0 | 0 |
| 40% | forced | 76 | 72.37% | +2.63 pp | 4 | 2 |
| 40% | forced_baseline_active | 35 | 74.29% | +2.86 pp | 1 | 0 |
| 40% | forced_baseline_inactive | 41 | 70.73% | +2.44 pp | 3 | 2 |
| 40% | suppressed | 115 | 67.83% | -5.22 pp | 0 | 6 |
| 40% | suppressed_baseline_active | 29 | 55.17% | -20.69 pp | 0 | 6 |
| 40% | suppressed_baseline_inactive | 86 | 72.09% | +0.00 pp | 0 | 0 |
| 50% | forced | 96 | 73.96% | +3.12 pp | 6 | 3 |
| 50% | forced_baseline_active | 43 | 76.74% | +4.65 pp | 3 | 1 |
| 50% | forced_baseline_inactive | 53 | 71.70% | +1.89 pp | 3 | 2 |
| 50% | suppressed | 95 | 68.42% | -4.21 pp | 0 | 4 |
| 50% | suppressed_baseline_active | 21 | 57.14% | -19.05 pp | 0 | 4 |
| 50% | suppressed_baseline_inactive | 74 | 71.62% | +0.00 pp | 0 | 0 |
| 60% | forced | 115 | 69.57% | +0.87 pp | 6 | 5 |
| 60% | forced_baseline_active | 46 | 76.09% | +2.17 pp | 3 | 2 |
| 60% | forced_baseline_inactive | 69 | 65.22% | +0.00 pp | 3 | 3 |
| 60% | suppressed | 76 | 71.05% | -5.26 pp | 0 | 4 |
| 60% | suppressed_baseline_active | 18 | 50.00% | -22.22 pp | 0 | 4 |
| 60% | suppressed_baseline_inactive | 58 | 77.59% | +0.00 pp | 0 | 0 |
| 70% | forced | 134 | 70.15% | +0.75 pp | 6 | 5 |
| 70% | forced_baseline_active | 51 | 76.47% | +3.92 pp | 3 | 1 |
| 70% | forced_baseline_inactive | 83 | 66.27% | -1.20 pp | 3 | 4 |
| 70% | suppressed | 57 | 71.93% | -5.26 pp | 0 | 3 |
| 70% | suppressed_baseline_active | 13 | 53.85% | -23.08 pp | 0 | 3 |
| 70% | suppressed_baseline_inactive | 44 | 77.27% | +0.00 pp | 0 | 0 |
| 80% | forced | 153 | 68.63% | -1.31 pp | 6 | 8 |
| 80% | forced_baseline_active | 56 | 75.00% | +1.79 pp | 3 | 2 |
| 80% | forced_baseline_inactive | 97 | 64.95% | -3.09 pp | 3 | 6 |
| 80% | suppressed | 38 | 73.68% | -5.26 pp | 0 | 2 |
| 80% | suppressed_baseline_active | 8 | 50.00% | -25.00 pp | 0 | 2 |
| 80% | suppressed_baseline_inactive | 30 | 80.00% | +0.00 pp | 0 | 0 |
| 90% | forced | 172 | 68.60% | -2.33 pp | 7 | 11 |
| 90% | forced_baseline_active | 62 | 72.58% | +0.00 pp | 3 | 3 |
| 90% | forced_baseline_inactive | 110 | 66.36% | -3.64 pp | 4 | 8 |
| 90% | suppressed | 19 | 73.68% | -5.26 pp | 0 | 1 |
| 90% | suppressed_baseline_active | 2 | 50.00% | -50.00 pp | 0 | 1 |
| 90% | suppressed_baseline_inactive | 17 | 76.47% | +0.00 pp | 0 | 0 |
| 100% | forced | 191 | 70.16% | -1.57 pp | 8 | 11 |
| 100% | forced_baseline_active | 64 | 76.56% | +3.12 pp | 3 | 1 |
| 100% | forced_baseline_inactive | 127 | 66.93% | -3.94 pp | 5 | 10 |
| 100% | suppressed | 0 | nan% | +nan pp | 0 | 0 |
| 100% | suppressed_baseline_active | 0 | nan% | +nan pp | 0 | 0 |
| 100% | suppressed_baseline_inactive | 0 | nan% | +nan pp | 0 | 0 |

## Reproducibility

- Shared X=10 forced outputs matching X=100: 19/19; generation drift 0.
- Shared suppressed outputs matching across X=10 and X=100: 0/0; generation drift 0.
- Additional X=100 policy-switch group: 172 samples, 53 naturally active at baseline, paired X=100 vs X=10 accuracy delta +2.91 pp.

## Interpretation

The controlled result is mixed or non-monotonic; one assignment seed is insufficient for a guidance change.
