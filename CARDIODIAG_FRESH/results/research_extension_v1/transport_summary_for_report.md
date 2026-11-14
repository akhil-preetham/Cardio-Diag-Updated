# Transport results and source dictionary for report

All results below are generated artifacts from the frozen, source-only shared-core transport protocol. No target-cohort tuning or probability updating was performed. These models are new five-core procedures, not the historical V3 or 38-field model. The executed source code and protocol are unchanged.

## Data dictionary and compatibility limits

| Predictor | Fixed mapping | Semantic status |
|---|---|---|
| Age | Numeric years; source-training median if missing; no clipping | Same named age field; timing unverified. Target has 6 missing and range21-89 vs source30-86. |
| Sex | Male=1; Female=0; local ZAS Fmale spelling=0 | Explicit text mapping; recorded sex, not a complete gender representation. |
| DM | Retain direct0/1 same-name field | Conventional diabetes indicator; exact provider ascertainment and target binary direction lack an accessible explicit codebook. No outcome-guided inversion. |
| HTN | Retain direct0/1 same-name field | Conventional hypertension indicator; exact ascertainment and target binary direction remain assumptions. |
| Current Smoker | Retain direct0/1; missing uses source-training mode | Target46missing; no missing->negative recoding before imputation. Definition/timing equivalence unresolved. |

Source: ZAS303rows,216CAD/87Normal, catalogue endpoint>=50% coronary narrowing. Target: official AngioCAD open CC BY4.0 workbooks,377clinical feature IDs matched to413label IDs;36label cases lack clinical records. Case IDs do not independently verify unique patients. Target diagnoses are STEMI186,NSTEMI20,UA55,SA116; non-obstructive controls are not necessarily healthy.

FH/EX-Smoker, source symptoms and ECG fields are unavailable in target. Outcome-defining artery fields, Gensini Score, Diagnosis, diagnostic series/echo fields, medications and uncertain-unit labs were excluded from the new predictor set.

## Prespecified estimators

Raw arm: L2 logistic regression with unbalanced loss; source-only3fold GridSearchCV overC=.1,1,10 using ROC-AUC; age imputation/scaling and binary mode imputation fitted inside training folds. Final model refits on all303source rows.

Calibrated arm: source-only3fold ensemble; each retained component performs its complete inner3fold C search within that component training subset, then fits a sigmoid on its disjoint source calibration subset. Target probabilities average the three retained source model/calibrator pairs. The arms differ in training size, averaging and calibration; differences cannot be attributed to calibration alone.

## Complete-repeat source OOF evaluation

| Arm | Mean ROC-AUC | Repeat range | Mean Brier |
|---|---:|---:|---:|
| raw_full_source_fit | 0.805884 | 0.802921-0.810398 | 0.152801 |
| source_calibrated_ensemble | 0.803933 | 0.801964-0.806513 | 0.153690 |

Each of three repeats predicts all303source rows once. Outer partitions are the first15historical frozen splits; repeat summaries reuse rows and are descriptive split variation, not independent patient samples or confirmatory population inference.

## Target endpoint sensitivity

| Endpoint assignment | Rows | Positive | Negative | Prevalence |
|---|---:|---:|---:|---:|
| base_gt50 | 377 | 298 | 79 | 0.790451 |
| exclude_ambiguous26_50 | 367 | 298 | 69 | 0.811989 |
| upper_assignment26_50_positive | 377 | 308 | 69 | 0.816976 |

Base rule is any segment51-75,76-90,91-99,or100 (>50%). Ten base-negative cases have a26-50segment. Removing them is an endpoint sensitivity subset; assigning them positive is an upper-bound sensitivity assignment, not a known >=50% label. Source>=50 versus target>50 remains an unresolved endpoint mismatch.

## Fixed-source target performance

| Endpoint | Arm | ROC-AUC | 95% record-bootstrap interval | AP | Brier | Log loss |
|---|---|---:|---|---:|---:|---:|
| base_gt50 | raw_full_source_fit | 0.712471 | 0.647437-0.773000 | 0.894134 | 0.198249 | 0.578727 |
| base_gt50 | source_calibrated_ensemble | 0.707225 | 0.645314-0.767955 | 0.896676 | 0.193659 | 0.562933 |
| exclude_ambiguous26_50 | raw_full_source_fit | 0.729282 | 0.662583-0.794264 | 0.913465 | 0.192493 | 0.560647 |
| exclude_ambiguous26_50 | source_calibrated_ensemble | 0.722717 | 0.656529-0.787618 | 0.914535 | 0.188079 | 0.547496 |
| upper_assignment26_50_positive | raw_full_source_fit | 0.725908 | 0.661931-0.792623 | 0.914984 | 0.194473 | 0.564747 |
| upper_assignment26_50_positive | source_calibrated_ensemble | 0.719462 | 0.656837-0.783950 | 0.916044 | 0.189543 | 0.550603 |

| Endpoint | Arm | Sensitivity at0.5 | Specificity at0.5 | Joint calibration intercept | Joint calibration slope |
|---|---|---:|---:|---:|---:|
| base_gt50 | raw_full_source_fit | 0.691275 | 0.632911 | 1.105168 | 0.553487 |
| base_gt50 | source_calibrated_ensemble | 0.731544 | 0.594937 | 1.075589 | 0.626857 |
| exclude_ambiguous26_50 | raw_full_source_fit | 0.691275 | 0.652174 | 1.249614 | 0.619421 |
| exclude_ambiguous26_50 | source_calibrated_ensemble | 0.731544 | 0.608696 | 1.215772 | 0.692256 |
| upper_assignment26_50_positive | raw_full_source_fit | 0.685065 | 0.652174 | 1.290822 | 0.608730 |
| upper_assignment26_50_positive | source_calibrated_ensemble | 0.724026 | 0.608696 | 1.257481 | 0.681082 |

Bootstrap intervals use2000target record resamples conditional on fixed source fits and the available target cohort; no source model refits, patient-cluster or centre-sampling uncertainty is included. Calibration metrics are a joint evaluation logistic regression on logit(p), not calibration-in-the-large with an offset, and were never applied to update predictions. AP must be interpreted against endpoint-specific prevalence. The0.5threshold has no clinical cost/utility justification.

## Simulated missingness stress (raw arm, no refitting)

| Per-cell MCAR probability | Mean source repeat ROC-AUC | Mean source repeat Brier |
|---:|---:|---:|
| 0.00 | 0.805883709 | 0.152800833 |
| 0.10 | 0.775782248 | 0.162595530 |
| 0.25 | 0.752164041 | 0.169814852 |
| 0.50 | 0.687544345 | 0.186714104 |

Masks are deterministic, independent of labels and nested across rates using one uniform draw per held-out cell. Source-training imputation is reused unchanged. This is simulated random missingness, not evidence about real MNAR clinical missingness or robustness in arbitrary hospitals. Exact masks and predictions are retained.

## Interpretation and QA

The five-core procedure has lower discrimination and worse probability performance in AngioCAD than in source OOF evaluation. Source-only calibration does not remove the transport gap. This is useful negative evidence about transport limits, not proof of reliable universal prediction. Source/target patient independence, clinical acquisition timing and complete binary-field semantic equivalence remain unverified.

Post hoc descriptive baseline context, not a prespecified selected arm: a constant probability equal to source prevalence216/303 gives target base-rule Brier0.171656859 and logloss0.529012098. Both transported models are worse on these proper scoring rules despite ranking AUC>0.7. The target-prevalence constant would be an apparent target-fitted benchmark and is not offered as an externally predicted model.

7focused transport tests passed after execution (including source-only calibration nesting, fixed mappings, endpoint ambiguity, imputation invariance, independent calibration arithmetic, bootstrap determinism and output/membership/hash checks). Full-style Ruff has5non-scientific findings (I001import ordering and UP017UTC alias); executed code remains untouched to preserve frozen hash. No claim of fully clean lint is made.

## Artifact sources

- Frozen protocol/hash: `transport/FROZEN_PROTOCOL.json` and `FROZEN_PROTOCOL.sha256` (SHA25661818ac6869acb2f3a0b5db50e6e187e24aa4d1cdeb69eb9ea87bd0ff05459a4).
- `transport/SOURCE_REPEAT_METRICS.csv`, `TARGET_METRICS.csv`, `TARGET_RECORD_BOOTSTRAP_INTERVALS.csv`, `SOURCE_MCAR_STRESS_REPEAT_METRICS.csv`.
- `transport/TARGET_PROBABILITIES_FROZEN_BEFORE_SCORING.csv`, `PREDICTION_FREEZE.json`, `SOURCE_MODELS.joblib` and `TRANSPORT_MANIFEST.json`.
- [ZAS provider](https://archive.ics.uci.edu/dataset/412/z+alizadeh+sani); [AngioCAD provider](https://zenodo.org/records/15826856); [AngioCAD paper](https://doi.org/10.1016/j.cmpb.2026.109368).
