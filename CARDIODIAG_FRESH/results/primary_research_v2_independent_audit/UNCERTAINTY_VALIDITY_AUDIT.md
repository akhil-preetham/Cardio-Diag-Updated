# Uncertainty and Robustness Validity Audit

## Classification: Correct with limitations

The bootstrap resamples the 303 source rows after averaging each row's repeated-CV predictions. It does not refit preprocessing, hyperparameters, estimators, or calibrators, and therefore captures conditional row-resampling variability only. It is not a population confidence interval, algorithmic uncertainty estimate, or patient-level bootstrap because no patient identifier is available.

The imbalance sensitivity output is descriptive and uses the same repeated-CV structure. The timing-feature sensitivity is explicitly not run. No individual predictive intervals are supported.
