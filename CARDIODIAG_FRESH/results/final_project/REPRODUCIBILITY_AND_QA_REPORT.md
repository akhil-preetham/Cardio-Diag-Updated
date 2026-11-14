# Reproducibility and QA report

## Read-only evidence reconciliation

- Phase 5 ZAS predictions: all 36 group/model combinations have 606 prediction rows, 303 source rows, two repeat predictions per row, and no label conflicts.
- Eligible-combined Logistic and Ridge: each reconstructs to ROC-AUC 0.928214, PR-AUC 0.966830, Brier 0.096424, and log loss 0.314108; maximum probability difference is zero.
- Hash manifests: Phase 5 40/40, Phase 4 multicentre 34/34, Phase 4.3 43/43 outputs present and matching recorded SHA-256 values.
- Python syntax: 115 source files parsed.
- Corrected UCI duplicate sensitivity uses `(centre, source_row)` and excludes only the two identified later duplicate occurrences; corrected pooled n=918. Original output files remain untouched.

## Automated checks

- Full test suite: 81 passed, 1 skipped.
- The skip is the optional LoRAS test because its dependency is unavailable. Two existing SAS readers were changed from invalid `encoding="infer"` to `encoding="utf-8"` so their tests pass.
- Ruff on the new `runner/final_project` utility: clean.
- Full Ruff on `src tests experiments`: 15 pre-existing lint findings remain, largely import sorting, unused imports, and unnecessary casts in older files. This is not a clean repository-wide lint result.
- LaTeX tools (`pdflatex`, `latexmk`, `bibtex`) are not installed/on PATH, so the manuscript could not be compiled. Citation keys were statically checked against the local bibliography.

## Limits

No model was fitted or expensive experiment rerun during this pass. The reconstruction utility reuses saved predictions and performs integrity/reconciliation work only. The repository and parent lack Git metadata, so no Git working-tree status or diff can be certified. Manifest checks establish file integrity relative to the manifests, not scientific correctness of every historical method.
