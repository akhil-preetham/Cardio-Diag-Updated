# Master evidence register

## Research question and evidence boundary

The consolidated question is: **“Retrospective discrimination of angiography-defined CAD using recorded clinical features.”** ZAS and UCI remain separate cohorts and tasks. No result establishes prospective prediction, verified patient-level independence, clinical utility, or external validation.

The metric-level inventory is in `MASTER_EVIDENCE_TABLE.csv`. Its rows are evidence records, not independent studies. Phase 4/4.3 artifacts overlap on the same 920 UCI records; Phase 5 and Phase 6 reuse the same 303 ZAS source rows. The Phase 5 AUC 0.928214 is a selected, post-hoc benchmark result. Phase 6 is the selection-aware ZAS primary analysis and was independently reconstructed in Phase 6.1.

## Predictor status

| Status | Evidence conclusion |
|---|---|
| Confirmed eligible | None: no predictor has source-confirmed acquisition time before the angiographic outcome. |
| Timing uncertain | All V3 and Phase 6 ordinary recorded predictors, including demographics, history, symptoms, ECG and laboratory fields. Preserve the original frozen V3 variable set as a reference, not as timing-confirmed. |
| Exploratory only | Expanded ZAS features and Phase 5/6 `eligible_combined` group, because temporal availability and ascertainment are undocumented. |
| Excluded due to leakage/post-diagnostic information | ZAS `Cath` is outcome only; EF-TTE, Region RWMA, VHD, Cath and angiographic fields are excluded from predictors. For UCI, `num` is target; `ca` and `thal` are excluded from stated stages. |

Do not promote timing-uncertain variables without source records or a codebook that verifies when they were collected. This inventory does not assert that every uncertain field was collected after diagnosis; it asserts that timing cannot be established from the available project evidence.

## Experiment register

| Track | What is supported | Limits and overlap |
|---|---|---|
| ZAS V3 | Frozen five-variable internal reference; 303 source rows; Cath Cad/Normal endpoint. | Source-row splits only; patient identity and predictor timing unverified. Phase 6 reports comparator repeat AUCs from reused frozen outer assignments. |
| ZAS Phase 5 | Saved predictions reconstruct the selected combined-group ROC-AUC 0.928214 and associated metrics. | Selected post hoc among 36 rows; Logistic/Ridge are aliases; not an unbiased final estimate. |
| ZAS Phase 6 | Selection-aware outer estimates exceed the paired V3 comparator in all three repeats; mean paired AUC difference +0.174001. | Only three repeats; no verified patient-level independence; timing uncertain; calibration scores are selection-dependent (Phase 6.1). A split-variation interval is not patient/population uncertainty. |
| UCI Phase 4/4.3 | Four-centre leave-one-centre-out staged performance is independently reconstructed on 920 records. | Same UCI cohort across reports/stages; Stage 2 and Stage 4 represent different information times/tasks; no patient-level clustering; not external validation of ZAS. |
| Phase 6.1 | Reconciles saved Phase 6 predictions, labels, fold/repeat metrics, comparisons, and manifest hashes; identifies calibration construction limitation and 4-vs-3 metadata error. | Audit of existing results, not a new experiment or calibration validation. |

## Source map

See the companion reconciliation and claims documents for the evidence ledger, manuscript-use guidance, and exact artifact paths. The current manuscript is a pre-Phase-6 draft and should not be represented as incorporating the Phase 6/6.1 evidence until separately revised.
