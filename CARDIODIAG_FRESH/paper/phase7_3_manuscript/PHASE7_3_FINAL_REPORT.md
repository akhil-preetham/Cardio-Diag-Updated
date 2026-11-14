# Phase 7.3 final report

## Decision

**MANUSCRIPT DRAFT COMPLETE — HUMAN DECLARATIONS PENDING**

A complete, connected Markdown journal manuscript has been assembled at `MANUSCRIPT.md` with title, authorship placeholders, structured abstract, introduction, related work, detailed methods, results, discussion, limitations, conclusion, declarations, data/code availability and numbered references. Tables and publication-quality figures are provided as separate editable/reusable artifacts and integrated into the manuscript. The Phase 5.1/6/6.1 evidence constraints and Phase 7.2 methodological interpretation are carried through the text.

## Scientific positioning

The primary ZAS result is the Phase 6 selection-aware comparison on three repeats of five paired outer source-row folds. The Phase 5 ROC-AUC 0.928214 is clearly separated as post-hoc exploratory evidence. The comparator is correctly described as a re-evaluated V3-style procedure. Predictor timing remains unconfirmed before the diagnostic decision; unique patient-level independence remains unverified. Calibration cross-fit scores are disclosed as selection-dependent, and the three-repeat split-variation interval is not presented as patient-level uncertainty or significance evidence. UCI is a separate four-centre staged evaluation, never an external validation of ZAS.

The novelty argument is intentionally narrow: a reproducibility and selection-aware reanalysis plus timing/interpretation audit of a small public benchmark, not a new classifier or a claim to be the first ZAS machine-learning study. Bibliographic metadata was checked against UCI, PubMed, journal/publisher records and DOI sources; the literature review is targeted and not systematic.

## Quality assurance

- Abstract and Table 3 Phase 6 repeat AUCs and paired differences matched `REPEAT_LEVEL_PERFORMANCE.csv` and `PAIRED_REPEAT_DIFFERENCES.csv` exactly before rounding.
- UCI pooled stage AUCs and Stage 3-to-4 paired estimates matched Phase 4.3 audited tables.
- Table cohort counts, event denominators and prevalence were checked against saved cohort/prediction artifacts.
- All numbered in-text citations 1–9 have matching reference records; image references resolve to generated PNG/PDF files.
- Existing Phase 4.3 and Phase 6 manifests still validate; Phase 7.1 and Phase 7.2 output hashes still match.
- No model training, hyperparameter search, benchmark rerun or historical-file modification was performed for Phase 7.3.
- No LaTeX compiler (`pdflatex`, `xelatex`, `lualatex`, `latexmk`) was available on the project PATH. The complete Markdown manuscript is supplied as the permitted alternative. PDF/typeset visual QA remains pending.

## Human declarations and remaining work

Author order/full names, affiliations, corresponding author, ethics/exemption determination, data-use permissions, funding, conflicts, contributions, patient/public involvement, protocol registration and persistent public code/data links remain author decisions. See `UNRESOLVED_DECLARATIONS.md`. The draft must not be submitted until these are resolved and an author-approved reporting checklist and final rendered layout are reviewed.

## Exact output namespace

All Phase 7.3 deliverables are under `paper/phase7_3_manuscript/`. `build_tables_figures.py` regenerates the tables/figures from saved results only. `REPRODUCIBILITY_NOTES.md` includes project environment details and the original experiment/audit commands, and makes clear that the model-fitting command was not run here.

Historical `paper/manuscript.tex`, Phase 1–7.2 artifacts and result namespaces were not edited.
