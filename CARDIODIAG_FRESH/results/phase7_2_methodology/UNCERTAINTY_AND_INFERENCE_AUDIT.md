# Uncertainty and inference audit

## Inventory

| Estimate/statement | Method and unit | n / resamples; overlap | What uncertainty represents | Permitted interpretation |
|---|---|---|---|---|
| Phase 6 repeat ROC-AUC and paired difference | Complete 303-row OOF set within each of 3 repeats; paired selection procedure minus V3 on same repeat | 3 repeat estimates; the same 303 source rows recur across repeats and their training partitions overlap | Sensitivity to the three selected split assignments | Report individual repeat results, mean and range descriptively. Not a patient-sampling interval. |
| Phase 6 “split-variation 95% CI” | Student t interval over three repeat-level metric differences; df=2 | n=3 repeated split summaries; not independent samples and not disjoint source cohorts | Estimated variation across this limited set of random partitions under the frozen protocol, with a very unstable variance estimate | Label exactly “descriptive split-variation t interval”; do not use p-values/significance or patient/population language. |
| Phase 6 outer-fold AUC SD/range | 15 outer-fold metric estimates | Each fold test set is disjoint within one repeat, but training sets overlap; folds across repeats reuse source rows | Fold composition/training-split variability; also fold prevalence/case-mix variation | Show fold plot/table as diagnostic instability, not 15 independent observations or CI. Fold AUCs are not interchangeable with complete-repeat OOF AUC. |
| Phase 4.3 paired stage intervals | Paired percentile bootstrap resampling of the same `(held_out_centre, source_row)` records with replacement; seed fixed in code | 400 bootstrap draws requested (`runner/phase4_validation_audit/run.py`, `BOOTSTRAPS=400`); unit is record; duplicate resampled records occur by design | Conditional record-level resampling uncertainty for the saved predictions and included four centres | State “record-bootstrap interval”; it does not capture centre sampling, patient clustering or model refitting. Do not infer clinical significance from interval overlap. |
| Phase 4.3 centre-specific metrics | Metrics within each held-out centre | Four centres; Switzerland n=123, 115 positive and 8 negative (93.5% event prevalence) | Descriptive centre performance; very sparse negative class yields unstable specificity/rank/calibration estimates | Present all four centres and denominators; avoid precise generalization to future centres. Switzerland AUC and specificity especially unstable. |
| Phase 4.3 pooled metrics | Pooled four-centre saved out-of-centre predictions | 920 source records; records are not verified patients; centre case mix differs | Performance on this particular four-centre mix; cross-centre ranking may contribute to pooled discrimination | Label pooled internal-external/leave-one-centre-out cohort summary, not external validation of ZAS or universal transport. |
| V3 `0.735 (SD 0.060)` | Historical fold summary; source specifies five outer folds/repeats as documented in V3 artifacts | Repeated source rows; not patient-level resampling | Variation among the historical split results | Descriptive baseline only; do not reinterpret SD as standard error or CI. Phase 6’s paired comparator uses its own matched re-evaluation. |

## Phase 6 numerical comparison

From `results/phase6_zas_selection/PAIRED_REPEAT_DIFFERENCES.csv` and `REPEAT_LEVEL_PERFORMANCE.csv`:

| Repeat | Selection-aware AUC | V3-style comparator AUC | Paired difference |
|---:|---:|---:|---:|
| 1 | 0.913421 | 0.737016 | +0.176405 |
| 2 | 0.899904 | 0.741965 | +0.157940 |
| 3 | 0.917625 | 0.729965 | +0.187660 |

Mean paired difference is +0.174001; observed range is +0.157940 to +0.187660. The t interval +0.136727 to +0.211276 is computed from only these three differences and estimates split variation, not uncertainty over patients. The analysis used 15 outer folds as training/evaluation operations, not 15 independent experimental samples. A formal null-hypothesis significance claim is not justified by treating repeats or folds as independent.

## Phase 4.3 paired record bootstrap

The runner requests 400 paired bootstrap resamples, resampling `n` row indices with replacement and applying the same sampled indices to both stage probabilities; degenerate single-class replicates are skipped for ROC-AUC. The audit report says comparisons use common `(held_out_centre, source_row)` records. This correctly preserves within-record pairing but does not preserve patient clusters (no verified patient key) or resample centres. The intervals therefore condition on the observed centres and do not quantify centre-to-centre sampling variability.

Reported Stage 3→4 differences: AUC +0.023528 [0.006084, 0.044258], AP +0.045381 [0.019600, 0.070624], Brier −0.009939 [−0.018608, −0.001509], log loss −0.027836 [−0.051392, −0.005624]. These are exploratory record-level intervals around saved OOF predictions; model-fitting uncertainty and multiplicity over stage comparisons are not captured. Do not state that the interval establishes clinical significance.

## Statistical language

- Prefer “was higher in each of the three evaluated repeats” over “significantly improved.”
- Report exact point estimates and fold/repeat or centre distributions. Keep ROC-AUC, average precision, Brier and log loss distinct.
- State class prevalence alongside AP (ZAS 0.7129). UCI AP is interpreted against its own cohort/centre prevalence, not compared directly with ZAS.
- Do not claim a patient-level confidence interval, population-level confidence interval, centre-generalization interval, or independent replication.

## Supporting artifacts

- Phase 6: `results/phase6_zas_selection/SELECTION_AWARE_PROTOCOL.md`, “Performance and uncertainty”; `REPEAT_LEVEL_PERFORMANCE.csv`; `PAIRED_REPEAT_DIFFERENCES.csv`; `PERFORMANCE_COMPARISON.csv`.
- Phase 6.1: `results/phase6_1_audit/RECONSTRUCTED_REPEAT_METRICS.csv`; `RECONSTRUCTED_PAIRED_REPEAT_DIFFERENCES.csv`; `PHASE6_METRIC_RECONCILIATION.md`.
- Phase 4.3: `runner/phase4_validation_audit/run.py`, lines 39 and 61–80; `results/phase4_validation_audit/PAIRED_STAGE_COMPARISONS.csv`; `PHASE4_3_FINAL_REPORT.md`; `CENTRE_SPECTRUM_ANALYSIS.csv`; `CENTRE_CALIBRATION_REPORT.md`.
- V3: `results/primary_research_v3/EXECUTIVE_RESULTS_REPORT.md` and `results/primary_research_v3_independent_audit/` reports.
