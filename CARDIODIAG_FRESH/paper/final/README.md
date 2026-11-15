# Final paper

`CARDIODIAG_FINAL_PAPER.pdf` is the current eight-page research report. `CARDIODIAG_FINAL_PAPER.tex` is its editable source. `FINAL_ANALYSES.pdf`/`.png` are the figure source; `PAPER_QA.json` records font, numeric and rendering checks. The original version-one and version-two reports remain in their own directories.

Formatting: Times New Roman, 10-point text, two columns. Mathematical symbols and superscripts use TeX math fonts. This is an IEEE-style project report with the user's typography customization; it does not assert compliance with a particular venue's current submission rules.

Build and verification instructions, results, release scope and remaining evidence limitations are in the root `FINALIZATION.md`. Licensed font files must be supplied locally and are excluded from Git.

## Reviewed submission-package integration

The current paper selectively integrates the supplied upgrade package's research questions, related-work discussion and claim boundaries, reconciled with the latest frozen analyses. The old snapshot's Latin Modern typography, hard-coded figure arrays and unsupported additional 1,500-draw bootstrap are not imported. No fitted model, default or experimental prediction changes.

- `SUPPLEMENTARY_METHODS.pdf` / `.tex`: five-page methods, literature comparison, exact paths, uncertainty scope and CSV-derived figures.
- `SUBMISSION_AUDIT.pdf` / `.tex`: two-page computational/editorial audit and author-confirmation items, without overlapping path tables.
- `NUMERIC_TRACEABILITY.csv`: 732 reconstructed metric values across 180 prediction groups, including all final analyses.
- `FIGURE_INPUTS.json`: frozen figure-input paths and hashes.
- `PAPER_EVIDENCE_MANIFEST.json`: code/data inputs and PDF/source/QA artifact hashes.
- `SUBMISSION_DOCUMENTS_QA.json`: font, page, bounds and visual-review checks.

Run `python tools/paper_evidence.py` from the repository root to validate the distributed artifacts and recompute metrics without training. Regeneration order is `paper_evidence.py --write`, `plot_final_analyses.py`, `build_final_paper.py`, then `build_submission_documents.py`; compile the three corresponding LaTeX sources using Tectonic. Render/review with `qa_final_paper.py` and `qa_submission_documents.py` in the document runtime, then refresh `paper_evidence.py --manifest`. All three documents use Times New Roman, 10-point text. An official journal template and unconfirmed author declarations remain separate submission requirements.
