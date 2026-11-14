# Final research decision

Date: 2026-11-13

## Decision

**Further experimental and reproducibility work is required before submission.** The current manuscript is a cautious, evidence-linked draft, not a submission-ready paper. This is a research-readiness decision, not a rejection of the observed exploratory results.

## What the evidence supports now

- The saved Phase 5 ZAS `eligible_combined` predictions reconstruct to ROC-AUC 0.928214, PR-AUC 0.966830, Brier 0.096424 and log loss 0.314108 across 303 source rows (216 positive labels).
- This was one result among 36 feature-group/model label combinations. Selection of the best result was post hoc and outside the nested model-tuning procedure. Logistic and Ridge are aliases in this implementation and have exactly identical predictions.
- Phase 4.3 has valid saved centre-held-out predictions for the four-centre UCI cohort. Its UCI endpoint and features do not constitute external validation of the ZAS task.
- A corrected duplicate sensitivity table has been reconstructed from saved predictions using centre-qualified row keys. It is a sensitivity analysis; historical outputs remain unchanged.

## Why submission is premature

- Acquisition time/order for symptoms, ECG and laboratory predictors is not documented at row level. Thus a defensible pre-angiography prediction-time task cannot be confirmed for the selected Phase 5 feature group.
- The Phase 5 feature search is exploratory and selection optimism is not estimated by an outermost selection layer or independent test set.
- No compatible independent validation cohort with the same endpoint, predictor definitions and timing evidence is established.
- The historical UCI duplicate sensitivity calculation is defective. The corrected reconstruction addresses the arithmetic/keying issue without changing historical outputs, but the correction should be reviewed and incorporated through a versioned protocol before making robustness claims.
- The local ZAS table has 56 columns including `Cath` (55 predictors), while the UCI dataset description reports 54 features. This schema discrepancy remains unresolved.
- Author, ethics/data-use, funding, contribution, patient/public involvement and availability statements are placeholders for human completion.

## Recommended next scientific decision

Freeze a single eligible feature set and target estimand using source documentation and clinical timing evidence; then run a versioned, fully nested selection protocol or obtain an independent test cohort. If row-level timing evidence cannot be recovered, frame the ZAS result as retrospective association/discrimination rather than deployable pre-test prediction. Do not promote 0.928 as a validated headline estimate.
