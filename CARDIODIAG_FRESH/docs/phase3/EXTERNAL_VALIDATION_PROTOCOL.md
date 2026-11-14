# External Validation Protocol

1. Freeze V3 logistic regression coefficients, preprocessing, calibration policy and threshold policy.
2. Confirm endpoint, predictors, units, timing, patient independence and missingness before opening labels.
3. Apply preprocessing and the frozen model without refitting.
4. Report ROC-AUC, average precision, Brier score, log loss, calibration intercept/slope, reliability plot, sensitivity, specificity, PPV, NPV and balanced accuracy.
5. Report uncertainty using patient-level bootstrap or appropriate analytic methods; do not treat repeated CV predictions as independent.
6. Perform subgroup analyses only when prespecified sample sizes support them.
7. Report untouched external performance before any recalibration.
8. If recalibration is later performed, label it as a separate model-updating experiment.
