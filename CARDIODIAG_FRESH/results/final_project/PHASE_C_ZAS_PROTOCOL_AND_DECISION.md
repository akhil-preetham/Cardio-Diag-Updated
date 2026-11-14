# Phase C — Prespecified ZAS evaluation protocol and decision

## Protocol freeze

**Reference:** the frozen primary-research V3 logistic baseline using Age, Sex, Current Smoker, EX-Smoker and FH, exact stored five-outer/four-inner split assignments, and fold-specific nested sigmoid calibration. No changes to the V3 protocol or outputs.

**Feature eligibility gate:** the source dataset has no acquisition timestamps. The feature matrix therefore marks the V3 variables B (plausible, timing unverified) and does not establish any category-A expanded candidate. The V3 result remains the historical reference because it was frozen by the project protocol; it is not relabelled as timing-confirmed.

**Candidate gate:** do not promote any expanded prediagnostic feature group. Phase 5's groups remain separate exploratory evaluations: basic history; symptoms/examination; resting ECG; labs; combined clinically plausible features; and the V3 feature set. Echo/downstream fields are excluded. The combined set's 38 inputs are not a source-verified prediagnostic schema.

**Models:** no new model training is justified while feature eligibility is unresolved. Preserve the prior bounded Phase 5 comparisons as exploratory. Correctly identify Logistic/Ridge as one estimator, with elastic-net logistic, spline logistic, random forest and histogram gradient boosting as the other label families. No winning model is selected from the table.

**Selection and uncertainty:** Phase 5 hyperparameter tuning is nested inside each candidate's outer training split, but group and model choice were not nested. Its fixed-prediction bootstrap is conditional on saved predictions and is not a model-development optimism correction. No patient-level bootstrap or independent cohort is available. Do not use the reported highest row as a prespecified final performance estimate.

## Observed results (descriptive only)

The `eligible_combined` logistic/ridge labels both have ROC-AUC 0.928214, PR-AUC 0.966830, Brier 0.096424 and log loss 0.314108 on 303 aggregated source rows. They are duplicate implementations and not independent candidates. Phase 5's V3-group logistic/ridge row is 0.742364 ROC-AUC, 0.856620 PR-AUC, 0.173249 Brier and 0.524220 log loss under the separate Phase 5 repeated 4×2 outer protocol. Those numbers are not directly comparable to V3's calibrated 5-fold result (ROC-AUC 0.735 ± 0.060), which uses a different protocol and calibrated probabilities.

The expanded groups have no verified timing basis and were compared after outcomes from all 36 rows were available. The apparent performance improvement is an exploratory association with the dataset's angiographic outcome, not evidence of earlier disease prediction or clinically meaningful improvement.

## Final ZAS decision

Retain V3 as the project reference and do not recommend an expanded final clinical model. The high-performing combined candidate is exploratory only. A defensible prediagnostic comparison requires source timing/ascertainment records and a prespecified candidate group before a new fit; an independent compatible cohort is required to establish external performance.
