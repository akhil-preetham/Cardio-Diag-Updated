# Statistical Analysis Plan v2

ROC-AUC is the primary discrimination metric. PR-AUC is a secondary ranking metric.
Balanced accuracy, sensitivity, specificity, precision, and F1 are threshold metrics
reported at a prespecified 0.5 threshold. Brier score, log loss, calibration intercept,
calibration slope, and reliability diagrams assess probability quality.

All model comparisons use identical outer test records. The preferred paper presentation
is descriptive paired differences with bootstrap intervals conditional on aggregated
out-of-fold predictions. Repeated-fold scores are dependent and must not be analyzed with
ordinary independent-samples t-tests or uncorrected Wilcoxon tests.

If an inferential comparison is needed, a corrected resampled procedure such as
Nadeau--Bengio may be considered as a sensitivity analysis. Its assumptions are fragile
under repeated five-fold CV and model-selection reuse; it must not be treated as definitive
population inference. Compare a prespecified limited set of models and state multiplicity.
No p-value is reported merely because it is below 0.05.

No threshold is optimized on outer test labels. Subgroup analyses are descriptive only
unless subgroup sizes and a prespecified estimand support more.
