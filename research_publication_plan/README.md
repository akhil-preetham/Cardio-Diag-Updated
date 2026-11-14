# CARDIODIAG research and publication package

Prepared 14 November 2026. This package is new work outside the historical research namespaces. No new predictive model was fitted.

Start with **RESEARCH_PLAN.md**: objectives, novelty, dataset selection, tuning/validation, compute choices, urgent versus stronger routes, and submission requirements.

| File | Purpose |
|---|---|
| BASE_VS_UPDATED.md | Base paper versus inspected original and updated implementation, with archive lineage checks |
| updated_methodology_review.md | Independent methodological review and exact calibration correction proposal |
| dataset_and_literature_review.md | Official online clinical dataset/source review and incompatibilities |
| data_screening/angiocad/SCREENING_REPORT.md | New open-data acquisition/screening; 377 matched clinical cases; no model results |
| IEEE_MANUSCRIPT_DRAFT.md | Editable, numbered-reference paper based on existing evidence |
| IEEE_MANUSCRIPT_DRAFT.tex | IEEEtran conference source; not compiled in this task |
| CARDIODIAG_IEEE_STYLE_REVIEW_DRAFT.pdf | Four-page two-column ReportLab review draft; not an official venue-validated export |
| MANUSCRIPT_NUMERIC_TRACEABILITY.csv | Means recomputed from existing repeat-level metric CSV; no training |
| BUILD_QA.json | Text/page/renderer checks and visual-review status |
| PACKAGE_MANIFEST.json | Hashes and final scope metadata |

The PDF is a readable IEEE-style draft, with placeholder authorship explicitly marked. Authorship/declarations, source/schema reconciliation, broader related-work review and a selected venue's requirements remain to be finalized. It does not claim completed new calibration, cross-cohort validation, clinical readiness or acceptance.

The most promising additional cohort is **AngioCAD**, DOI 10.5281/zenodo.15826856. Only the two small clinical/label workbooks and public metadata were downloaded, with provider checksum verification. No approximately 16 GB image archives were acquired. Stenosis boundary, missing fields and clinical spectrum prevent direct application of the historical 38-feature or V3 model. Treat a new shared-feature study as a separately frozen extension.

The fastest credible route uses current retrospective evidence and its disclosed limits. The first meaningful extra comparator is a regularized logistic model using the same rich feature set. A stronger follow-up can test selection-independent calibration and cross-cohort transport, with uncertain or negative findings retained.

Rebuild the review PDF/LaTeX/numeric table using `python build_manuscript.py` with ReportLab, pypdf, pypdfium2 and Pillow. This regenerates only this package, and resets visual QA to pending until the new page images are inspected. The LaTeX source requires IEEEtran and a LaTeX installation; a generated .tex file is not proof of compilation.
