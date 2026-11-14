# Manuscript results outline (evidence map only)

This is an outline for a later manuscript reconstruction. No manuscript files were edited in Phase 7.1.

1. **Cohorts and endpoints:** ZAS 303 source rows, 216 Cad/87 Normal; UCI 920 records, binary `num` endpoint. Emphasize separate cohorts/tasks and unverified patient identity.
2. **Feature eligibility:** report no source-confirmed acquisition times; list confirmed eligible as none, all candidate V3/expanded ZAS fields as timing uncertain, and downstream/target fields excluded. Avoid implying uncertain means known post-diagnostic.
3. **Frozen ZAS V3 reference:** present as historical internal source-row comparator only (five features; original reported AUC 0.735, SD 0.060). Clarify timing remains unverified.
4. **Phase 5 exploratory reconciliation:** optionally place AUC 0.928214/AP 0.966830/Brier 0.096424/log loss 0.314108 in a clearly labeled post-hoc exploratory paragraph or supplement; explain selection from 36 benchmark rows and Logistic/Ridge alias.
5. **Phase 6 primary ZAS result:** describe frozen candidate pool, nested outer/inner selection, three repeats × five paired outer folds, repeat-level AUC and paired differences versus V3. Show repeat variability; do not substitute the split-variation interval for patient-level uncertainty.
6. **Robustness and calibration:** group selection 15/15 and family counts; feature importance as exploratory. State calibration scores are selection-dependent, and calibration training-size mismatch remains. Avoid interpreting calibrated scores as validated probabilities.
7. **Separate UCI results:** report Phase 4.3 stages in their own table, preserving Stage 2 and Stage 4 task distinctions and paired Stage 3-to-4 increments. Identify record-bootstrap uncertainty and centre sparsity.
8. **Evidence conclusion:** supports only retrospective discrimination evidence; no prediagnostic, external validation, clinical utility, patient-level inference or deployment claims.

## Existing manuscript state

`paper/manuscript.tex` is a draft centered on Phase 5/UCI evidence and does not yet include Phase 6/6.1 as the primary ZAS analysis. Its results and methods require a separately approved update after review of this evidence package. References and figures were not altered; author, ethics, funding, conflicts and other submission declarations remain unresolved in the existing publication-readiness report.
