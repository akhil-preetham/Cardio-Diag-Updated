# Results reconciliation

## ZAS

The local endpoint is `Cath` (Cad=1, Normal=0), with 303 source rows, 216 positive and 87 negative labels (prevalence 0.712871). Patient identity is not verified.

Phase 5 selected `eligible_combined` Logistic/Ridge at ROC-AUC 0.928214 from 36 reported group/model rows. The labels Logistic and Ridge are the same L2 logistic estimator and yielded identical probabilities. This is a post-hoc selected result and is retained only as exploratory history.

Phase 6 evaluated a predeclared pool of six feature groups and five distinct model families within three repeats of five paired outer folds. The original Phase 5 AUC was not used to define the candidate pool. Repeat-level selection-aware / V3 values were:

| Repeat | AUC selection-aware | AUC V3 | Difference | AP selection-aware | AP V3 | Brier selection-aware | Brier V3 | Log loss selection-aware | Log loss V3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.913421 | 0.737016 | +0.176405 | 0.960211 | 0.852784 | 0.107522 | 0.174118 | 0.344735 | 0.525719 |
| 2 | 0.899904 | 0.741965 | +0.157940 | 0.953653 | 0.851938 | 0.113745 | 0.173406 | 0.366622 | 0.525106 |
| 3 | 0.917625 | 0.729965 | +0.187660 | 0.958010 | 0.835864 | 0.101582 | 0.176291 | 0.328864 | 0.534874 |

Each repeat contains all 303 source rows out-of-fold; these repeat sets are not independent patient samples. Prevalence was 0.712871 in each repeat, which is the reference level for interpreting average precision. The mean paired ROC-AUC difference is +0.174001; the reported +0.136727 to +0.211276 interval describes variation across three split differences only.

Selection-aware group frequency was `eligible_combined` 15/15. Selected family counts were L2 logistic 9/15, elastic net 4/15, spline logistic 2/15. This group consistency does not demonstrate individual feature necessity. Typical Chest Pain had the largest reported marginal permutation importance, which is not causal and may reflect symptom ascertainment or work-up. Timing of the selected features remains unresolved.

Phase 6.1 independently verified 1,818 prediction rows (303 × 3 repeats × 2 procedures), labels, partitions, selected metadata, fold metrics, repeat metrics and paired arithmetic. It found no material discrimination discrepancy. Calibration is more limited: Phase 6 selected groups/hyperparameters using all outer-training labels before generating inner out-of-fold scores used for sigmoid fitting and threshold selection. Calibration scores are therefore selection-dependent; inner calibration fits use about two-thirds of outer training data while final refit uses all of it. This affects calibration interpretation, not the independent outer-test discrimination estimate. Calibration intercept/slope are assessed on outer-test predictions; outer-test labels did not select the candidate or calibration procedure.

## UCI

The UCI four-centre cohort comprises 920 records and binary `num` outcome (0 vs 1–4), a distinct task from ZAS. Phase 4.3 reports pooled leave-one-centre-out metrics:

| Stage | ROC-AUC | PR-AUC/AP | Brier | Log loss |
|---|---:|---:|---:|---:|
| Stage 1 basic risk | 0.717900 | 0.729290 | 0.211205 | 0.611668 |
| Stage 2 symptoms | 0.822525 | 0.824077 | 0.168070 | 0.510644 |
| Stage 3 resting measures | 0.821505 | 0.819088 | 0.168381 | 0.513922 |
| Stage 4 exercise test | 0.845033 | 0.864469 | 0.158443 | 0.486086 |

Paired Stage 3-to-4 record-level changes: AUC +0.023528 (95% record-bootstrap interval +0.006084 to +0.044258), AP +0.045381 (+0.019600 to +0.070623), Brier −0.009939 (−0.018608 to −0.001509), log loss −0.027836 (−0.051392 to −0.005624). These are paired source records keyed by centre and source row, not patient/centre-cluster inference. Centre-specific performance and calibration are unstable in sparse strata. Phase 4/4.3 reports and stages overlap and must not be counted as independent validations. UCI is not external validation of ZAS.

## Reconciliation sources

- `results/phase5_1_audit/ZAS_0928_EVIDENCE_DECISION.md`
- `results/phase6_zas_selection/REPEAT_LEVEL_PERFORMANCE.csv`
- `results/phase6_zas_selection/PAIRED_REPEAT_DIFFERENCES.csv`
- `results/phase6_zas_selection/ROBUSTNESS_REPORT.md`
- `results/phase6_1_audit/PHASE6_METRIC_RECONCILIATION.md`
- `results/phase6_1_audit/PHASE6_CALIBRATION_AUDIT.md`
- `results/phase6_1_audit/PHASE6_1_FINAL_DECISION.md`
- `results/phase4_validation_audit/PHASE4_3_FINAL_REPORT.md`
- `results/phase4_validation_audit/PAIRED_STAGE_COMPARISON.csv`

Values are transcribed from saved audited artifacts; this consolidation did not refit models or bootstrap data.
