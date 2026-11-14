# Phase 6 robustness report

## Selection stability

The selection-aware procedure selected `eligible_combined` in 15/15 outer fits. Group counts: eligible_combined 15/15. Model-family counts: logistic_l2 9/15, elastic_net 4/15, spline_logistic 2/15. The best candidate group was consistent across the tested outer partitions; the selected model family varied.

All 15 outer fits selected the same 38-feature group. Thus group inclusion frequency is 15/15 for each feature in that group and zero for features in unselected groups; it is not evidence that each feature is individually necessary. See `FEATURE_SELECTION_FREQUENCY.csv` for the complete per-feature counts. A separate marginal permutation assessment is below.

## Outer-fold variability and feature dependence

Across 15 outer folds, selection-aware ROC-AUC mean=0.914, SD=0.046, range=0.837–0.980. V3 mean=0.739, SD=0.071, range=0.637–0.858. The full fold-level metrics are in `PERFORMANCE_COMPARISON.csv`.

Outer-test marginal permutation importance was computed only after each model was frozen and never fed back into selection. Highest mean AUC decreases:

- Typical Chest Pain: mean held-out marginal AUC decrease 0.125 (SD 0.050; present in 15 selected-group folds).
- Age: mean held-out marginal AUC decrease 0.042 (SD 0.017; present in 15 selected-group folds).
- DM: mean held-out marginal AUC decrease 0.019 (SD 0.011; present in 15 selected-group folds).
- Tinversion: mean held-out marginal AUC decrease 0.015 (SD 0.010; present in 15 selected-group folds).
- HTN: mean held-out marginal AUC decrease 0.012 (SD 0.009; present in 15 selected-group folds).
- TG: mean held-out marginal AUC decrease 0.012 (SD 0.009; present in 15 selected-group folds).
- FH: mean held-out marginal AUC decrease 0.010 (SD 0.007; present in 15 selected-group folds).
- PR: mean held-out marginal AUC decrease 0.006 (SD 0.008; present in 15 selected-group folds).
- St Elevation: mean held-out marginal AUC decrease 0.005 (SD 0.006; present in 15 selected-group folds).
- Q Wave: mean held-out marginal AUC decrease 0.004 (SD 0.004; present in 15 selected-group folds).

This suggests discrimination depends most on Typical Chest Pain, with Age and DM next in this marginal analysis. Permutation importance is not causal, correlated features can mask one another, and fold test samples are small.

## Sensitivity to uncertain feature exclusions

The separate Age/Sex-only model (both timing-uncertain, not confirmed eligible) had repeat-level ROC-AUC 0.729 (range 0.726–0.731); repeat-level Brier 0.179. This approaches the V3 baseline and is materially below the full selected procedure. No confirmed-eligible-only model can be run because the local evidence confirms no predictor acquisition timestamps. Excluding all uncertain features therefore leaves no scientifically defensible predictor set for this endpoint.

## Paired comparison and uncertainty

Selection-aware minus V3 repeat-level ROC-AUC differences were +0.176, +0.158, +0.188 (mean +0.174; descriptive split-variation 95% t interval +0.137 to +0.211). All 15 paired outer-fold AUC differences were positive in 15/15 folds (min +0.088, max +0.276). These intervals summarize only variation across the selected three split repeats and are not patient-level or population confidence intervals. Repeated predictions are not treated as independent people; no record bootstrap or centre-level inference was used.

PR-AUC is average precision. Event prevalence was 0.713 in every repeat, and that prevalence is the PR reference in the result tables. Threshold-based sensitivity and specificity use fold-specific thresholds chosen on inner out-of-fold predictions.

## Interpretation boundary

All Phase 5 model features remain timing-uncertain. The result is retrospective discrimination of angiography-defined CAD from recorded variables, with no verified patient-level independence. Typical Chest Pain carries the largest marginal importance and could reflect symptom ascertainment/referral or diagnostic work-up. This analysis does not support prediagnostic availability, clinical utility, prospective prediction, or external validation.
