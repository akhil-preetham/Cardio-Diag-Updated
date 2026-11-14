# Leakage-Controlled Clinical Machine Learning for CAD Identification

## Title

Leakage-Controlled, Interpretable Internal Validation of Clinical Machine Learning for
Coronary Artery Disease Identification

## Keywords

Coronary artery disease; clinical machine learning; nested cross-validation; calibration;
reliability; explainability; data leakage

## Abstract

This draft reports an internal-validation study of coronary artery disease identification
in the Z-Alizadeh Sani cohort. Five strictly pre-diagnostic features were evaluated using
five repeats of stratified five-fold nested cross-validation. The highest descriptive mean
calibrated ROC-AUC was 0.737 (SD 0.059) for
`logistic_regression_none`. Its aggregated repeated-CV bootstrap ROC-AUC was
0.738 (95% interval
0.677-0.790).
These results are internal validation only and do not establish clinical utility or
external validity.

## Introduction

Coronary artery disease classifiers can appear highly accurate when diagnostic-procedure
variables or contaminated validation procedures are included. This study evaluates a
strictly pre-diagnostic feature set and explicitly separates discrimination from
probability reliability.

## Related Work

The project audit documented historical preprocessing and validation risks. This draft
does not claim a systematic literature review or add unverified citations; verified
dataset and methodological references remain to be inserted before submission.

## Materials and Methods

### Dataset Description

The study used 303 Z-Alizadeh Sani records, with 216 CAD and 87 normal labels. The
dataset is a single clinical cohort and is not an external validation sample.

### Data Preprocessing

The frozen predictors were Age, Sex, Current Smoker, EX-Smoker, and FH. Numeric values
used median imputation and standardization; Sex used most-frequent imputation and
one-hot encoding. These transformations were learned inside each training pipeline.

### Leakage-Controlled Validation

Five repeated stratified five-fold outer splits and four-fold inner selection were used.
Every row was evaluated only in its outer test fold for each repeat. Cleveland and
Statlog were not used as external validation because their records overlap exactly.

### Model Development

Logistic regression, support vector classification, random forest, HistGradientBoosting,
and Extra Trees were evaluated. Class-weighted variants were compared where supported.
No SMOTE was used.

### Calibration and Reliability

Uncalibrated predictions were compared with sigmoid probabilities learned by
`CalibratedClassifierCV` on outer-training data and inner folds only. Brier score,
log loss, calibration intercept and slope, and ten-bin expected calibration error were
reported.

### Explainability

Permutation importance was computed on each outer test partition using the fitted
uncalibrated pipeline. These values describe predictive association and not causation.

### Statistical Analysis

The primary metric was ROC-AUC. PR-AUC, balanced accuracy, sensitivity, specificity,
precision, F1, Brier score, and log loss were secondary. Paired observation-bootstrap
effects were reported for model comparisons; independent-fold significance tests were not
used because repeated CV folds are dependent.

## Results

The numerical results below are generated from the saved artifacts:

- Best calibrated mean ROC-AUC: 0.737 (SD 0.059).
- Best-model aggregated bootstrap ROC-AUC: 0.738
  (95% interval 0.677--0.790).
- All model-fold fits completed successfully.

## Discussion

The strict feature set produced moderate discrimination and is substantially less
optimistic than historical results that used broader or methodologically compromised
tracks. The result supports continued investigation, not a clinical performance claim.
Calibration and model ranking remain uncertain in this small cohort.

## Limitations

The cohort is small and single-source; feature timing is incompletely documented; repeated
CV predictions are dependent; and no independent, target-compatible external cohort was
available. Cleveland/Statlog overlap prevents their use as independent validation.

## Conclusion

Strict feature eligibility and nested validation materially constrain the defensible
interpretation of CARDIODIAG results. The saved artifacts provide a reproducible internal
baseline for future, independently validated research.

## References

Dataset and methodological references are pending verification from the project provenance
audit. No unverified citation is included in this draft.

## Methods

The target was `Cath`: CAD (`Cad`) versus normal (`Normal`). The frozen primary feature
set was Age, Sex, Current Smoker, EX-Smoker, and FH. `Cath`, EF-TTE, Region RWMA, and VHD
were excluded because of target or downstream diagnostic leakage risk. Numeric imputation
and scaling, categorical imputation and one-hot encoding, hyperparameter selection, and
sigmoid calibration were performed within training partitions.

The model set comprised logistic regression, support vector classification, random forest,
HistGradientBoosting, and Extra Trees, with class-weighted variants where supported.
ROC-AUC was primary; PR-AUC, operating-point metrics, Brier score, log loss, calibration,
permutation importance, and paired bootstrap comparisons were secondary.

## Research Traceability

Results are traceable to `results/primary_research_v1/`. No prospective outcome, clinical
utility analysis, ethics statement, funding statement, author contribution statement, or
journal submission claim is made. Citation and author metadata remain pending and must be
completed from verified sources before submission.
