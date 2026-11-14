# Proposed Research Protocol

## Primary question
Can a prespecified model using genuinely pre-diagnostic clinical variables estimate CAD presence in the Z-Alizadeh Sani cohort without diagnostic-procedure leakage, with uncertainty and calibration reported transparently?

## Primary dataset and target
Use Z-Alizadeh Sani (`Cath`: Cad versus Normal). Exclude `Cath`, `EF-TTE`, `Region RWMA`, and `VHD` from the primary predictor set. ECG, symptoms, functional class, and examination findings require a documented prediction-time definition before inclusion. Use only demographics, history, baseline measurements, and laboratory variables confirmed available before catheterization.

## Validation
Use repeated nested stratified cross-validation with an outer patient-level split and an inner tuning split. Fit imputation, scaling, encoding, feature selection, and any resampling within inner training folds only. Do not use repeated overlapping holdouts as external validation. Reserve a genuinely independent cohort only after provenance and target compatibility are established; current Cleveland, Statlog, Five States, Mendeley, and Kaggle relationships do not meet that standard.

## Missingness and imbalance
Report missingness before any handling. Prespecify clinically justified imputation within training folds, add missingness indicators only if justified, and perform sensitivity analyses. If class weighting or resampling is used, apply it inside training folds only and compare against an unresampled baseline.

## Evaluation and uncertainty
Report discrimination, calibration, threshold-independent and clinically relevant threshold metrics with nested resampling uncertainty intervals. Use bootstrap or repeated outer-fold intervals, not test-set tuning. Compare models with paired outer-fold predictions only after the protocol is frozen. Decision-curve analysis is exploratory and is not evidence of clinical utility without an explicit net-benefit context.

## Secondary tracks
Treat NHANES as complex-survey descriptive/associational analysis, PTB-XL/PTB Diagnostic as separate patient-level ECG tasks, Auricle as multimodal physiology metadata until a storage decision, and UCI Heart Failure as a separate mortality task. Do not merge targets or claim cross-track clinical generalization.
