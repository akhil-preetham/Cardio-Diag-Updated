# Feature timing and clinical claims review

## Timing classification under a source-evidence standard

“Before the Cath outcome” is weaker than “available before the diagnostic decision.” The current repository has no row-level timestamps or source collection records that establish this distinction. Accordingly, no predictor is confirmed available before the diagnostic decision in the strict sense requested here.

| Classification | Variables / groups | Finding and evidence |
|---|---|---|
| Confirmed available before diagnostic decision | None established from source-time evidence. | Phase 6 protocol says no predictor is timestamp-confirmed; `results/phase6_zas_selection/SELECTION_AWARE_PROTOCOL.md`, “Feature eligibility and candidate pool.” |
| Timing uncertain | Age, Sex, Current Smoker, EX-Smoker, FH; DM, HTN; symptom and examination features; resting ECG; laboratory fields. | Phase 5.1.1-named audit file is not present; available audit is `results/phase5_1_audit/ZAS_FEATURE_TIMING_AUDIT.md` plus `ZAS_FEATURE_TIMING_TABLE.csv`. It calls Age/Sex category A because inherently knowable before the Cath outcome, but admits exact recording time is undocumented. That does not prove presence in the record before the diagnostic decision; therefore retain uncertainty for the stricter claim. The remaining included predictors are explicitly timing-uncertain. |
| Potentially downstream or post-diagnostic | EF-TTE, Region RWMA, VHD; some Function Class, ECG, laboratory or work-up values could have been collected after referral or during diagnostic work-up, but this is unverified. | `ZAS_FEATURE_TIMING_AUDIT.md`, “Result traced” and “Classification”; `ZAS_FEATURE_TIMING_TABLE.csv` entries. Do not assert actual post-diagnostic collection without source timestamps. EF-TTE/Region RWMA/VHD were excluded and are described as downstream risks. |
| Target/direct outcome information | Cath; angiographic narrowing/vessel fields and other target-derived fields. | Cath mapped to Cad=1/Normal=0 at `runner/phase5/zas_track/run.py`; excluded from predictors. See `results/phase5_1_audit/ZAS_LEAKAGE_AUDIT.md` and `ZAS_FEATURE_TIMING_AUDIT.md`. |

The Phase 5.1.1 label in the user request does not match an artifact namespace found in the repository. The evidence available is under `results/phase5_1_audit/`; no separate Phase 5.1.1 audit was identified. This is a naming/evidence traceability limitation, not evidence that a timing audit was never performed.

## Evidence needed to resolve timing

For each feature, obtain the original codebook/data dictionary and collection protocol; confirm the feature definition and source system; obtain encounter-level measurement timestamps or a validated workflow sequence relative to referral, diagnostic testing, angiography and outcome adjudication; document whether retrospective abstractions were updated after diagnosis; and verify provenance with dataset maintainers or source-site records. Deidentified public data without timestamps may make this permanently unresolved.

## Clinical interpretation

Frame the ZAS result exclusively as **retrospective discrimination of angiography-defined CAD using recorded clinical features**. The endpoint is established by angiography and the data do not establish feature availability at an earlier clinical decision. Typical Chest Pain’s high marginal permutation importance is descriptive and may reflect ascertainment/referral; it is not causal evidence or a usable prediagnostic signal. See `results/phase6_zas_selection/ROBUSTNESS_REPORT.md`, “Outer-fold variability and feature dependence” and “Interpretation boundary.”

Do not claim prospective prediction, risk prediction before angiography, early detection, clinical utility, clinical readiness, safe triage, or a timestamp-confirmed eligible model.
