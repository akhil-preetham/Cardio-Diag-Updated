# Correction Recommendations

The existing primary results are preserved and are not silently replaced.

1. **Future experiment metadata:** save the exact inner-fold assignments, not only the seed formula.
2. **SVM probability labeling:** rename or separately document `expit(decision_function)` as a score-derived probability scale; do not compare its Brier/log-loss values directly with native probabilistic estimators without qualification.
3. **Calibration design:** use a separately nested calibration layer or pre-specified calibration procedure if probability quality is a confirmatory objective.
4. **ECE:** reduce the number of bins or use a pre-specified adaptive/bin-count sensitivity analysis; retain ten-bin ECE only as descriptive for these small outer folds.
5. **Uncertainty:** if inferential claims are needed, use a design that accounts for algorithm refitting and split variation, and reserve population inference for an independently sampled cohort.
6. **Statistical comparison:** retain the current paired bootstrap as descriptive; do not add p-values without a defensible dependent-CV comparison method and multiplicity plan.
7. **Clinical validity:** retain the five-feature primary analysis. Any broader exploratory set must document timing and units before modeling and must never be merged with the primary estimate.

No correction is required to the existing outer-fold predictions or reported primary metric recomputation.
