# Publication-readiness report

## Status

**Further experimental and reproducibility work required before submission.** A scientifically cautious manuscript draft and supplementary package exist. This status reflects unresolved feature timing, post-hoc selection among 36 ZAS candidates, no compatible external validation, an invalid historical duplicate-sensitivity implementation (with a separate corrected reconstruction now available), and author-dependent declarations. The manuscript could not be compiled because a LaTeX toolchain is unavailable.

## Evidence ready for reporting

- ZAS 0.928214 ROC-AUC reconstructed from saved predictions; Logistic/Ridge exact alias documented; result classified exploratory.
- Frozen V3 five-feature configuration and nested-calibration protocol retained as reference, with internal-validation limitation.
- UCI Stage 1–4 pooled and centre-specific metrics summarized, paired comparisons keyed by centre/source row, Stage 2 vs Stage 4 prediction-time distinction preserved.
- Corrected UCI duplicate sensitivity reconstructed separately using `(centre, source_row)`; historical artifacts remain unchanged.
- External validation explicitly unavailable for frozen V3.
- Data timing, lack of verified patient identifiers, Swiss class imbalance, centre heterogeneity, historical duplicate sensitivity defect and uncertainty boundaries disclosed.
- Relevant source references and citation keys were statically checked against the local bibliography.

## Reproducibility checks completed

- Saved prediction reconciliation and artifact checks: see `results/final_project/REPRODUCIBILITY_AND_QA_REPORT.md` and `SAVED_ARTIFACTS_QA.json`.
- Full tests: 81 passed, 1 skipped (optional LoRAS dependency unavailable).
- Full Ruff: 15 pre-existing findings remain; the new audit utility passes Ruff.
- Python syntax: 115 files parsed. No models were refit.
- Manifest hashes matched for Phase 5 (40/40), Phase 4 multicentre (34/34), and Phase 4.3 (43/43).

## Required before submission

1. Recover acquisition-time evidence for each predictor and resolve ZAS schema discrepancy (local CSV has 55 predictors plus target; UCI metadata reports 54 features).
2. Freeze the target estimand and feature eligibility; if timing cannot be confirmed, retain retrospective exploratory framing.
3. Review corrected UCI duplicate sensitivity and issue any correction as a new version without replacing historical outputs.
4. Run a selection-aware outer validation or obtain a compatible independent validation cohort before making a performance claim for a selected ZAS model.
5. Confirm data permissions and stable public code/data artifact links.
6. Complete authorship, affiliations, ethics/consent determination, funding, conflicts, contribution, patient/public involvement and corresponding-author details.
7. Compile and visually inspect the manuscript with a LaTeX toolchain, then complete the reporting checklist and clinical/statistical review.

## Submission boundary

The manuscript is an evidence audit and secondary analysis of saved results. It is not a validated clinical prediction model, clinical utility evaluation or external validation study. A reporting checklist does not certify methodological quality.
