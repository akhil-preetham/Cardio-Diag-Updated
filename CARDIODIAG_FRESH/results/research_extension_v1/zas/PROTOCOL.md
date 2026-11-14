# ZAS retrospective research extension v1 — frozen protocol

Dated 14 November 2026. This namespace is a new, retrospectively motivated follow-up after inspecting Phase 5/6 results. It is not prospectively independent confirmation. The full user-authorized protocol runs all 15 historical outer partitions even if performance decreases. Historical predictions and files must remain unchanged. Source/protocol/data/split/historical-prediction hashes are frozen in FROZEN_INPUTS.json before fitting; checkpoint reuse requires those exact hashes. No checkpoints can be silently replaced.

## Objective and data

Retrospective discrimination of angiography-defined Cath CAD on 303 ZAS source rows (216 Cad, 87 Normal). No verified independent-patient or predecision timing claim is made. Same local source, six timing-uncertain groups, all five historical family implementations, all original grids and dtype-based preprocessing are retained for exact comparability. No clinical semantic typing revision is introduced into the historical candidate pool. Numeric-coded categories may enter numeric/spline branches; this is a retained limitation. Downstream fields Cath, EF-TTE, Region RWMA and VHD remain excluded from predictors. The installed scikit-learn API interprets interior l1_ratio with saga as elastic net; preserve exact environment/version evidence.

## Splits and seed policy

Reuse first three repeats of the existing five-fold outer manifest: 15 paired partitions. Legacy inner folds inside each outer train are StratifiedKFold(3, shuffle=True, seed=61051+100*repeat+fold). Retain source-row IDs in membership traces. The same three blocks are calibration blocks C_k for the corrected ensemble; D_k is the other two blocks. For a component's selection, new three-fold CV is created inside D_k with seed=861051+1000*repeat+100*fold+k. No outer evaluation targets or target column are passed to training APIs. The same source rows have been investigated previously: changing the training construction does not recover independent external validation.

## Arm 1: equal-feature L2 control

Use all38 eligible_combined fields and the exact ordinary L2 logistic pipeline from historical make_pipeline (lbfgs, no class weighting, C in [.1,1,10]). Select C on the same legacy inner folds. Generate crossfit probabilities after C selection, fit the exact historical C=1e6 lbfgs sigmoid on raw probabilities (not logits), select Youden J with the original tie rule, retain the GridSearchCV refitted full outer-training estimator, and apply its calibration map to outer test. This deliberately preserves historical selection dependence/training-size mismatch to make a fair *information-matched procedural control*. Compare against saved historical selection predictions on identical outer partitions. No claim that new independent calibration was used in this control.

## Arm 2: selection-independent three-component calibration ensemble

For each outer-training calibration block C_k:

1. Search the complete 6 groups x5 families (30 candidates, 120 hyperparameter settings, original ordering/tie rule) with three-fold inner CV exclusively on D_k. All transforms fitted within D_k inner fits. C_k labels cannot choose the base pipeline or its parameters.
2. Use the selected GridSearchCV refit on D_k to score C_k. Fit the same C=1e6 raw-probability sigmoid there. C=1e6 is fixed for comparability; it is weak regularization, not evidence of a well-regularized small calibration model.
3. Keep that fitted D_k base model and C_k calibrator together, never refitting the base model with C_k. Obtain outer-test raw and calibrated scores from the same retained pair.
4. Average the three component calibrated probabilities. A paired uncalibrated arm averages raw probabilities from those exact components. Only this calibrated-versus-identical-raw contrast isolates the calibration operation. Comparing the corrected ensemble to old single-refit selection also changes training size, averaging and selection instances, so it is NOT a pure calibration-effect estimate.

No outer labels affect modeling. Calibration block sizes (~80) are small and sigmoid coefficients may be unstable. Finite output, convergence warnings and performance declines are reported. Fixed 0.5 thresholds are descriptive only for corrected arms; no screening threshold or utility claim. No calibrator choice or regularization tuning after outer scores are seen.

## Arm 3: information-group ablations

Use ordinary L2 with C [.1,1,10] and the same historical pipeline/calibration/Youden rules for all four information sets: full38; no-symptoms28 (drop all10 symptoms_examination fields); risk+symptoms17 (seven risk plus ten symptoms/examination); ordinary V3-five (the same five features but ordinary unweighted L2). Reuse full38 arm, do not duplicate fitting. This separates feature availability at fixed family/preprocessing/calibration construction. The ordinary V3-five ablation is not identical to the separately retained historical balanced-V3 comparator; identify both explicitly. All feature timing remains uncertain.

## Outputs and comparisons

Persist every component/control's selection, all candidate-best scores, all parameter-grid inner scores, membership traces, held calibration score/label pairs, sigmoid coefficients, probabilities, warning counts/messages, wall time and package versions in immutable JSON checkpoints. No stored classifier unpickling is needed to audit predictions.

Final outputs: row-level PREDICTIONS, FOLD_METRICS, REPEAT_METRICS, SUMMARY_METRICS, PAIRED_REPEAT_DIFFERENCES, PAIRED_DESCRIPTIVE_SUMMARY, SELECTIONS, CANDIDATE_SCORES, GRID_SCORES, SPLIT_TRACE, WARNINGS, RUNTIMES and hashed MANIFEST. Import historical selection and balanced-V3 probabilities as explicitly historical reference arms; do not refit or overwrite them.

Primary comparisons: historical selection vs equal-feature38 L2 (complete-repeat AUC); corrected calibrated ensemble vs identical raw-component ensemble (Brier); full38 L2 vs three information ablations (AUC/Brier). AP, log loss, sensitivity/specificity and joint logit calibration intercept/slope are secondary descriptive metrics. Calibration diagnostic intercept here is not slope-fixed calibration-in-the-large. The existing original-fold mean is not substituted for complete-repeat metrics. Report mean, SD, min/max and all three paired repeat differences; no inferential t-tests, population confidence intervals or claims based on 15 independent folds. No new seed/feature/model choices based on scores.

## Compute and verification

Limit BLAS/OpenMP and GridSearchCV/RF jobs to1. Log first complete outer partition as a runtime pilot, then complete the entire locked 15-partition design automatically under the user's unlimited-time authorization. Per-component search is ~390 base fits:45 components imply ~17,550 fits before controls. Runtime is measured, not promised. Resource failures are reported and resume must validate source hashes; no outcome-based stopping.

Tests independently verify all source labels and one-prediction-per-row/repeat/arm coverage, development/calibration/test disjointness, nested inner membership, selected candidate rank, same-component averaging, recalculation of final metrics, historical hashes and the design's lack of outer-target training arguments. No data leakage, patient independence or clinical reliability is inferred merely from passing software tests.
