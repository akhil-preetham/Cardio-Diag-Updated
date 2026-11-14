# Phase 6 selection-aware ZAS protocol (frozen before outer evaluation)

## Research question and estimand

**Retrospective discrimination of angiography-defined CAD using recorded clinical features.** The target remains `Cath` (Cad=1, Normal=0). The unit is a dataset source row because no verified patient identifiers exist; patient-level independence and patient-cluster uncertainty remain unestablished. No prospective prediction or confirmed prediagnostic availability claim is made.

## Feature eligibility and candidate pool

The local evidence supports no predictor as timestamp-confirmed eligible. Fifty-two non-target features are timing-uncertain (category B); three diagnostic/downstream features (EF-TTE, Region RWMA, VHD) are exploratory-only/excluded from candidate models (category C); `Cath` is the outcome only (category D). No predictor is promoted to confirmed eligibility. Candidate features are only the exact Phase 5 groups listed in the runner: V3 five predictors, basic risk, symptoms/examination, resting ECG, laboratory, and eligible_combined. All are explicitly **timing-uncertain retrospective** feature sets. The demographic-only Age/Sex model is a separate, prespecified sensitivity analysis and is not called confirmed-eligible.

The candidate model families are the five unique Phase 5 implementations: L2 logistic, elastic-net logistic, spline logistic, random forest, and histogram gradient boosting. `ridge` is dropped because it is the same L2 LogisticRegression implementation as `logistic` with the same grid. Phase 5 grids are frozen in `runner/phase6_zas_selection/run.py`; no candidate is added after outer evaluation begins. The comparison baseline is the V3 report's highest descriptive model, logistic regression with class weighting on the five V3 predictors and inner-selected C, evaluated again on the same outer partitions. For Phase 6, its features/family/class weighting are fixed, while C, sigmoid calibration and the Youden-J threshold are fit within each outer training partition under the same evaluation rules as the selected procedure. It is a re-evaluated V3 reference comparator, not an untouched independent validation or the exact historical fitted equation/0.5-threshold rule.

## Split and nested selection

Reuse the first three repeats of the frozen Stage 6/V3 manifest: three repeats of five outer folds (15 outer partitions), paired between the selection procedure and V3. Generate three deterministic stratified inner folds within each outer training partition (seed = 61051 + 100 × repeat + fold). This bounded alternative was chosen after a full 25-outer-fold run exceeded 35 minutes of CPU without producing outputs. It retains the original outer assignments for paired comparison while reducing inner-fold computation. The row assignments are stable source-row assignments, not patient groups.

For each outer fold:

1. Compare all 30 frozen feature-group × unique-family candidates using only the three inner validation folds; each candidate's hyperparameters are selected using ROC-AUC on those same inner folds. The maximum mean inner ROC-AUC selects the group, family and parameter set, with ties resolved by the declared candidate order.
2. With the winner and parameters fixed from the full inner search, refit that selected pipeline on each inner-training partition and predict its held inner fold. These out-of-fold predictions train the sigmoid calibration map and select the Youden-J threshold. The calibration fit is part of the training procedure; the untouched outer fold evaluates any calibration optimism introduced by winner selection on the inner folds.
3. Fit a sigmoid calibration map to the three selected-procedure out-of-fold scores. Separately repeat this V3-style cross-fitting for the locked V3 logistic-balanced comparator.
4. Choose a Youden-J threshold from the calibrated inner out-of-fold scores, using a deterministic highest-threshold tie rule.
5. Refit the full-inner-selected candidate (and V3 comparator) on all outer-training rows, calibrate its outer-test probabilities, and evaluate exactly once on the outer test fold.

All imputation, one-hot encoding, scaling, spline expansion and model fitting are in sklearn pipelines fit only on each training partition. The outer test fold is not used for group, model, hyperparameter, calibration or threshold selection. The outer predictions estimate the full selection procedure. The base candidate and parameters are held fixed across calibration folds within an outer fit. V3 and the selected procedure use this same Phase 6 calibration operation so the comparison is on matched outer partitions with a consistent evaluation procedure; V3's historically reported 0.735 is not reused as Phase 6 validation.

## Performance and uncertainty

Report per-outer-fold metrics and three repeat-level metrics. Each repeat contains one out-of-fold prediction per source row. The repeat-level mean, SD, range and t interval summarize **split variation only**; they are not patient-level confidence intervals because folds/repeats reuse source rows and patient identity is unknown. Do not use an ordinary record bootstrap. Paired selection-minus-V3 differences are computed within each repeat and summarized descriptively across three repeats with the same split-variation caveat.

PR-AUC is average precision and is reported alongside the event prevalence baseline for every fold/repeat. Calibration intercept/slope are diagnostic regression estimates on each outer test partition; small folds make them unstable. Sensitivity and specificity use thresholds chosen only from inner training predictions. No metric establishes clinical utility.

## Robustness

- Selection frequencies are tallied across the 15 outer fits for feature group and model family; selected-group inclusion frequencies are reported per predictor.
- Fold-level and repeat-level variability are reported rather than hidden in one pooled score.
- A separate Age/Sex-only sensitivity excludes all non-demographic Phase 5 predictors. It remains timing-uncertain and cannot substitute for a confirmed-eligible model.
- Marginal permutation importance is computed on outer test folds only after the model is frozen, is never used for selection, and is descriptive. It can identify concentration of discrimination in a few features but does not provide causal interpretation.
- The published `.928` Phase 5 result is not consulted by the runner's selection logic.

## Exact reproduction commands

From `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH` in PowerShell:

```powershell
$env:PYTHONPATH = 'src;.'
.\.venv\Scripts\python.exe -m runner.phase6_zas_selection.run
.\.venv\Scripts\python.exe -m pytest tests\test_phase6_zas_selection.py -q
.\.venv\Scripts\python.exe -m runner.phase6_zas_selection.finalize
```

The runner refuses to write into a non-empty `results/phase6_zas_selection/` except for this frozen protocol file. To reproduce after completion, copy/archive that namespace and run in a clean checkout. Do not run Phase 5 or Phase 5.1 commands as part of Phase 6 reproduction.


