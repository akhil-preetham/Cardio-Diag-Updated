# Manuscript consistency audit — initial pass

Inspected paper/phase7_4_1_final_manuscript/MANUSCRIPT.md against Phase 6/6.1 code, tables, decisions and focused tests. This is not an exhaustive claim-by-claim external bibliographic verification or a fresh rendered-page inspection.

| Claim/issue | Assessment |
|---|---|
| Repeat AUCs and paired differences | Consistent with inspected saved table and passing metric-reconciliation tests |
| Calibration selection dependence and training-size mismatch | Explicitly disclosed; no claim of correction |
| V3 comparator | Correctly described as matched re-evaluation with inner C tuning |
| Patient unit, uncertainty and timing | Correctly bounded to source rows, descriptive split variation and uncertain timing |
| UCI | Clearly separate; no ZAS external-validation claim |
| Phase 5 0.928 | Explicitly exploratory/post hoc |
| Author list | Corrected manuscript lists only Ch Hari Krishna and G Vennela as names awaiting confirmation; user handover additionally lists Akhil Preetham Erukudinnela. Human reconciliation required; do not infer final order |
| Environment wording | Manuscript says working requirements are not fully pinned; later QA documents a frozen snapshot. Explain distinction in a future isolated revision, without implying original requirements were locked |
| Candidate prespecification | Source asserts frozen pool, and outer tests remained outside operational selection; independence from all prior dataset exploration is not established by within-run nesting alone |
| Bibliography and source relationships | Not freshly verified online in this pass; no new claim of citation correctness |
| Layout | Historical ReportLab 12-page QA report exists; LaTeX compilation unverified |
| College report | Current content audit describes a six-page paper-style output. Full chapter-based institutional report requirements remain a separate task |

Do not rewrite historical manuscript files during audit. Any later corrections should use a new version and traceable manifest.
