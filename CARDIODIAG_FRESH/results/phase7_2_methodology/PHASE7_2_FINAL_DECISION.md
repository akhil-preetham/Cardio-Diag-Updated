# Phase 7.2 final decision

## Classification

**METHODOLOGICALLY DEFENSIBLE WITH LIMITATIONS** for a carefully bounded retrospective discrimination manuscript after specified reporting corrections.

The Phase 6 outer evaluation is appropriately paired on the same frozen ZAS source-row partitions; candidate groups, unique model families and tuning grids are frozen; preprocessing is pipeline-local; and outer-test labels do not select candidates, tune parameters, fit calibration or choose thresholds. The saved complete-repeat OOF metrics and paired differences were independently reconciled in Phase 6.1. The defensible conclusion is descriptive: the selection-aware procedure had higher repeat-level ROC-AUC than the matched, re-evaluated V3-style comparator in each of three evaluated repeats.

This does not establish patient-level or population-level superiority. Three overlapping source-row repeats do not provide independent experimental replicates; the reported t interval is split-variation only, and no statistical significance claim is justified. Calibration crossfit scores are selection-dependent, final model training size differs from calibration-fold training size, and Youden thresholds have no clinical interpretation. These limitations constrain calibration and threshold claims but do not create an outer-test label leakage path that invalidates outer discrimination.

The paper must be framed as retrospective discrimination of angiography-defined CAD using recorded clinical features. No feature has source-confirmed availability before the diagnostic decision. UCI is a separate four-centre staged evaluation, not external validation of ZAS. The Phase 5 AUC 0.928214 is exploratory and post hoc.

## Confirmed defects

- Phase 6.1 found the calibration-selection metadata reports four folds for an operational three-fold inner design; reporting-only, no effect on fitting or predictions (`results/phase6_1_audit/PHASE6_IMPLEMENTATION_AUDIT.md`).
- The older timing audit labels Age/Sex available before the outcome although exact recording time is undocumented; that does not establish availability before the diagnostic decision (`results/phase5_1_audit/ZAS_FEATURE_TIMING_AUDIT.md`, lines 11–16; `ZAS_FEATURE_TIMING_TABLE.csv`). Use the stricter “timing uncertain” classification for manuscript claims.
- Refer to the Phase 6 baseline as a re-evaluated V3-style comparator, not the exact historical frozen fitted model/threshold (`results/phase6_zas_selection/SELECTION_AWARE_PROTOCOL.md`, “Feature eligibility and candidate pool”).

## Limitations, not defects invalidating discrimination

- Source-row evaluation with unverified unique patient identity.
- Three outer repeats and no patient-level uncertainty.
- Selection-dependent calibration crossfit, training-size mismatch and unstable fold calibration.
- No source evidence for predecision feature timing.
- UCI record-level bootstrap, four observed centres only, sparse Switzerland negatives, and distinct task/outcome schema.
- No independent compatible ZAS external cohort or clinical utility evaluation.

## Unresolved human decisions

Author/affiliation and contribution details; ethics/data permissions; funding/conflicts; availability of source timing evidence; target journal; and whether the authors accept the retrospective-only framing. No manuscript edits or new experiments are part of this audit.

## Scope completion

No model fitting, hyperparameter search or benchmark was run. No Phase 1–7.1 artifact was modified. New review outputs are confined to `results/phase7_2_methodology/`.
