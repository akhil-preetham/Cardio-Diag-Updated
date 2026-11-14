# Phase 6 metric reconciliation

## Verification result

The saved outer predictions were independently matched to the original 303-row ZAS dataset and the frozen outer split manifest for repeats 1–3. There are 15 paired outer partitions and 1,818 saved prediction rows (303 rows × 3 repeats × 2 models). Every source row appears once in each repeat/model set, each partition matches the manifest test rows exactly, `source_row_id` matches the row index without asserting patient identity, and every saved `y_true` matches `Cath` mapped as Cad=1/Normal=0.

The selected feature group/model metadata matches every row in the corresponding outer fold. Reconstructed fold metrics are in `RECONSTRUCTED_OUTER_FOLD_METRICS.csv`; all ROC-AUC, average-precision PR-AUC, Brier, log loss, threshold-based sensitivity/specificity, and calibration intercept/slope values reconcile to `PERFORMANCE_COMPARISON.csv` within 1e-10. Every row has n=60 or 61. Average precision is shown with each fold's observed prevalence reference.

## Repeat-level reconstruction

Each repeat was rebuilt as one complete 303-source-row out-of-fold set before computing ROC-AUC and the other metrics. Results reconcile to `REPEAT_LEVEL_PERFORMANCE.csv`; paired differences reconcile to `PAIRED_REPEAT_DIFFERENCES.csv`.

| Model | Mean of repeat-level AUCs | Mean of 15 fold AUCs (diagnostic only) |
|---|---:|---:|
| Selection-aware procedure | 0.910316 | 0.913887 |
| Phase 6 V3 reference comparator | 0.736315 | 0.739266 |

The unweighted fold-AUC average is not substituted for the AUC computed across a repeat's full 303 out-of-fold predictions. The paired repeat-level ROC-AUC differences (selection minus V3 comparator) are +0.176405, +0.157940, +0.187660; their mean is +0.174001. This reconciles with the rounded Phase 6 decision claim.

## Conclusion

**Outer discrimination is verified from the saved predictions.** No metric discrepancy was found. This does not establish patient-level independence or population-level uncertainty; rows are the only verified split unit.
