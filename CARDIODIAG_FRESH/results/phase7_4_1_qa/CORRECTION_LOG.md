# Phase 7.4.1 correction log

| Area | Change | Evidence / scope |
|---|---|---|
| Manuscript preservation | Copied the Phase 7.3 manuscript package to `paper/phase7_4_1_final_manuscript/`; edits and PDF are confined to that versioned copy. | All 46 input and 28 output hashes in the original Phase 7.3 manifest still match. No Phase 1–7.3 historical result was regenerated. |
| Figure 2 source | Added a figure-only generator using saved Phase 6 repeat-level metrics; explicit 0–1 ROC-AUC axis, 0.2 ticks, metric label and 300-dpi PNG; writes PNG and PDF from one source. | Original figure source explicitly used `.65–1.0`; the corrected full scale does not alter data. Original PNG/PDF retained in Phase 7.3 and copied as `*_ORIGINAL` sidecars. |
| Figure 2 caption | Clarified complete 0–1 y-scale and retained the warning that repeat points overlap and are not independent patient samples. | Manuscript Markdown in the corrected copy only. |
| Reproducibility notes | Added isolated clean-environment and figure-only regeneration instructions, and documented that PDF rendering uses a separate local ReportLab runtime. | No model training, benchmark or Phase 6 rerun. |
| PDF renderer | Added a Markdown-to-PDF review renderer and a read-only page/text raster check. Tables with at least 10 columns use landscape pages; relative PDF sidecar paths are printed visibly rather than emitted as local-file links. | Final render: 12 portrait pages, all pages rasterized, no blank text pages/replacement glyphs/missing expected sections; visual review found no clipping or overflow. |
| Clean environment | Created `.venv-phase7_4_1`, installed the Phase 7.4 pinned snapshot, matched all 126 pinned packages, ran `pip check` and four focused smoke tests. | Existing `.venv` and `.venv-repro` were left untouched. No training/search. |
| Reporting appraisal | Added a complete itemized TRIPOD+AI reporting checklist and PROBAST+AI evidence/applicability assessment. | Missing items and items requiring author confirmation remain explicitly labeled; population applicability and predictor timing remain unresolved. |

The copied `PHASE7_3_FINAL_REPORT.md` and `PHASE7_3_MANIFEST.json` are retained unchanged inside the corrected package as historical provenance copies. They continue to describe Phase 7.3; the Phase 7.4.1 manifest in the results namespace is the integrity manifest for this correction package.

No values in saved experiment results were changed. No UCI or ZAS experiment was run. Git status was unavailable because this checkout contains no Git metadata.
