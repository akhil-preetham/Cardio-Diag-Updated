# Comprehensive CARDIODIAG Research Assessment

## Executive decision

The scientifically preferred internal-validation baseline remains V3 logistic regression with fully nested sigmoid calibration and the five strict predictors: Age, Sex, Current Smoker, EX-Smoker and FH. V4 compact optimization did not improve V3. Exploratory feature groups showed apparent performance gains, especially symptoms, ECG, laboratory and medical-history groups, but they are not final eligible predictors because prediction-time timing, ascertainment and/or units are insufficiently documented.

## Dataset and generalization decision

Z-Alizadeh Sani remains the primary cohort for the angiographic CAD question. Cleveland is not an untouched external validation cohort for the strict model because the required predictors are unavailable in a harmonized form; Cleveland and Statlog also have confirmed record-level overlap. Other datasets are incompatible or separate research tracks. No external validation result is claimed.

## Methodological decision

V3 corrects the V2 calibration-selection interaction. V4 preserved that architecture and independently passed split, calibration isolation, prediction completeness, metric recomputation and hash checks. Repeated-fold results remain dependent, bootstrap summaries remain conditional, and no confirmatory superiority claim is made.

## Publication decision

The package is a reproducible internal-validation and methodological case study, not a clinically deployable model. Before journal submission as a clinical prediction study, the project needs governance documentation, source-level clinical timing and unit confirmation, and an independent compatible validation cohort.
