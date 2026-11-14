# CARDIODIAG Research Protocol v2

## Status and scope

This is the protocol for the next experiment, not a replacement of Version 1 results.
The primary task is small-cohort CAD classification in 303 Z-Alizadeh Sani records:
`Cath = Cad` versus `Cath = Normal`. It is a restricted demographic/risk-factor study,
not a comprehensive clinical diagnostic system.

## Primary features

The frozen primary set remains `Age`, `Sex`, `Current Smoker`, `EX-Smoker`, and `FH`.
No feature is added because it improves ROC-AUC. The clinical audit found these five
unambiguously eligible, although source documentation should still confirm prediction-time
availability. Timing-sensitive or unit-uncertain features are exploratory only.

## Leakage prevention

All imputation, scaling, encoding, feature selection, resampling, hyperparameter
selection, and calibration are fit within training partitions. Exact outer and inner
assignments must be saved. Source-row identifiers are reproducibility labels only; they
are not patient identifiers. No external dataset is used as validation.

## Validation and models

Use five prespecified repeated stratified outer folds and four inner folds, with exact
seeds and assignments persisted. Use a compact set of logistic regression, SVM, random
forest, HistGradientBoosting, and Extra Trees models. Compare class weighting only where
the estimator supports it. Avoid SMOTE unless a separate protocol justifies it.

## Probability and uncertainty

Native probability outputs and calibrated outputs must be labeled separately. SVM scores
must not be treated as probabilities without a nested calibration procedure. Primary
probability evaluation uses held-out Brier score, log loss, calibration intercept/slope,
and reliability plots; ECE is supplementary and sensitivity-tested. Uncertainty is
reported as split/repeat variation and conditional resampling summaries, never as
unqualified population confidence intervals.

## Statistical comparison

The primary paper reports paired descriptive differences on identical outer predictions.
Dependent-CV methods such as corrected resampled tests may be reported only as sensitivity
analyses with assumptions and multiplicity stated. No ordinary independent-fold tests are
allowed.

## Explainability and acceptance

Held-out permutation importance and logistic coefficients are predictive associations,
not causal effects. Results are accepted only when split manifests validate, all
predictions are complete, no failures are hidden, metrics recompute, hashes match, and
limitations are reported.
