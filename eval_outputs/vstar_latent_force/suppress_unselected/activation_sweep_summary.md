# VStarBench activation sweep summary

Chart: `activation_sweep_chart.png`

| Target X | Realized activation | Accuracy | Delta vs natural | Corrected | Broken |
|---:|---:|---:|---:|---:|---:|
| natural | 64/191 (33.51%) | 71.73% | +0.00 pp | 0 | 0 |
| 10% | 19/191 (9.95%) | 67.54% | -4.19 pp | 2 | 10 |
| 20% | 38/191 (19.90%) | 67.02% | -4.71 pp | 2 | 11 |
| 40% | 76/191 (39.79%) | 69.63% | -2.09 pp | 4 | 8 |
| 50% | 96/191 (50.26%) | 71.20% | -0.52 pp | 6 | 7 |
| 60% | 115/191 (60.21%) | 70.16% | -1.57 pp | 6 | 9 |
| 70% | 134/191 (70.16%) | 70.68% | -1.05 pp | 6 | 8 |
| 80% | 153/191 (80.10%) | 69.63% | -2.09 pp | 6 | 10 |
| 90% | 172/191 (90.05%) | 69.11% | -2.62 pp | 7 | 12 |
| 100% | 191/191 (100.00%) | 70.16% | -1.57 pp | 8 | 11 |

## Reading

- Natural activation is 64/191 (33.51%) with 71.73% accuracy.
- The best controlled point is X=50 with 71.20% accuracy, still 0.52 pp below natural.
- X=10 and X=20 are clearly worse because many naturally active samples are suppressed.
- Pushing activation above 50% does not create a monotonic gain; X=80/90 drop more.
- Forcing baseline-inactive samples is mixed, while suppressing baseline-active samples is consistently harmful.
