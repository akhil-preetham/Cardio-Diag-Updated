# Calibration and threshold review

## Findings

1. **Selection dependence is confirmed.** `choose_candidate()` first selects group, family and hyperparameters using labels from all three inner folds. `crossfit_selected_scores()` then withholds each inner fold to generate calibration scores, but holds the already selected group/parameters fixed. Labels in a held calibration block have therefore already influenced candidate/hyperparameter selection. The scores are estimator-out-of-fold, not selection-procedure-out-of-fold. The sigmoid and threshold are fitted on these scores and labels. See `runner/phase6_zas_selection/run.py`, lines 149–188 and 190–207; `results/phase6_1_audit/PHASE6_CALIBRATION_AUDIT.md`, “Calibration limitations.”

2. **No outer-test calibration leakage was found.** Outer test labels are not passed to selection, preprocessing, sigmoid fitting or threshold selection. They are used for evaluation after predictions are fixed. Phase 6.1 independently recomputed calibration intercept/slope from outer-test predictions. Therefore calibration limitations do not invalidate outer discrimination. See `PHASE6_CALIBRATION_AUDIT.md`, “Outer test use and validity.”

3. **Training sizes differ.** Inner score-producing estimators train on roughly 2/3 of outer-training rows (example 161/161/162 fit rows); the final outer estimator trains on all 242 outer-training rows. A change in fit size can alter the raw score scale and therefore sigmoid transport. The outer-test evaluation captures performance of this construction on held-out rows, but it is not calibration validation for a final model refit on all 303 rows. See `PHASE6_CALIBRATION_AUDIT.md`, “Calibration limitations.”

4. **Brier/log loss are outer-test probability scores, but their interpretation is constrained.** They combine discrimination and calibration and are estimates for the full procedure on the observed source-row split regime. Due to selection-dependent calibration crossfits and training-size mismatch, describe them as exploratory/limited probability performance, not validated individual risk probabilities. Calibration intercept/slope are computed from outer-test predictions and are unstable on folds of about 60 rows. See `results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv`, `CALIBRATION_RESULTS.csv`, and Phase 6.1 calibration audit.

5. **Threshold is statistical, not clinical.** Youden J maximizes sensitivity + specificity − 1 on the same inner OOF calibrated scores. It assigns equal implicit weight to these two rates and ignores prevalence, harms, treatment consequences, resource constraints and decision-curve utility. Threshold-based sensitivity/specificity are secondary descriptive metrics only. The fold-specific thresholds differ; their mean is not a globally chosen operating point. See `runner/phase6_zas_selection/run.py`, `choose_threshold()` lines 198–207 and `REPEAT_LEVEL_PERFORMANCE.csv` (`mean_selected_threshold`).

6. **Metadata count is a confirmed reporting-only error.** A four-entry calibration-selection annotation was emitted for a three-fold process. Phase 6.1 established it does not affect fits or predictions. Do not “correct” Phase 6 historical output in place; cite the Phase 6.1 corrected metadata audit. See `results/phase6_1_audit/PHASE6_IMPLEMENTATION_AUDIT.md`.

## Publication recommendation

For a retrospective discrimination paper, retain outer-test Brier/log loss and calibration intercept/slope only as secondary, explicitly limited probability-assessment measures; show fold and repeat variation and the baseline prevalence. Present threshold metrics as exploratory and do not describe the Youden threshold as clinically selected. A separate, versioned nested calibration design is **not necessary to support the narrow discrimination claim**, since outer tests remained outside all selection/calibration decisions. It **is necessary** before making strong claims about calibrated risk, a fixed final model, or any clinically actionable threshold. No historical results should be overwritten.

## Supporting artifacts

- `runner/phase6_zas_selection/run.py`: `choose_candidate`, `crossfit_selected_scores`, `fit_sigmoid`, `choose_threshold`, and outer prediction block.
- `results/phase6_1_audit/PHASE6_CALIBRATION_AUDIT.md`.
- `results/phase6_1_audit/CALIBRATION_TRAINING_SIZE_AUDIT.csv`.
- `results/phase6_1_audit/OUTER_CALIBRATION_METRIC_CHECKS.csv`.
- `results/phase6_1_audit/PHASE6_IMPLEMENTATION_AUDIT.md`.

