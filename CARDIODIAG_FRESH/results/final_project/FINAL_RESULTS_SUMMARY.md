# Final results summary

## ZAS Phase 5 exploratory result

| Item | Reconciled value |
|---|---:|
| Source rows | 303 |
| Positive target labels | 216 |
| Outer prediction rows per candidate | 606 |
| Eligible-combined Logistic ROC-AUC | 0.928214 |
| PR-AUC | 0.966830 |
| Brier score | 0.096424 |
| Log loss | 0.314108 |

Metrics were recomputed after averaging the two saved repeat predictions for each source row. The values match the benchmark. This is an exploratory selected result among 36 benchmark combinations, not an unbiased estimate for the best selected candidate. Logistic and Ridge outputs are exact aliases, not independent confirmations.

## UCI four-centre stage results

| Stage | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Stage 1 basic risk | 0.717900 | 0.729290 | 0.211205 | 0.611668 |
| Stage 2 symptoms | 0.822525 | 0.824077 | 0.168070 | 0.510644 |
| Stage 3 resting measurements | 0.821505 | 0.819088 | 0.168381 | 0.513922 |
| Stage 4 exercise test | 0.845033 | 0.864469 | 0.158443 | 0.486086 |

These are saved out-of-centre predictions, not external validation of the ZAS model. The corrected sensitivity analysis excluding only the two later within-centre duplicate occurrences has n=918; pooled AUCs are 0.716972, 0.822549, 0.821332 and 0.844978 for Stages 1–4 respectively. The correction is stored separately and does not overwrite the historical Phase 4.3 outputs.

## Frozen V3 reference

The original ZAS V3 reference uses Age, Sex, Current Smoker, EX-Smoker and FH with 5 outer folds, 4 inner folds, and fold-specific nested sigmoid calibration. Its recorded highest descriptive calibrated ROC-AUC is 0.735 ± 0.060. This is an internal reference protocol and has no verified patient identifier or row-level acquisition timestamps.
