# Final CARDIODIAG research report

The final PDF is **8 pages**, compiled from the IEEEtran conference class using Tectonic 0.17.0. The Markdown and LaTeX sources are editable. All eight rendered pages were visually inspected. Four grayscale figures and four tables describe completed experiments, not projected scores.

Confirmed author order:

1. Erukudinnela Akhil Preetham
2. Chunduri Hari Krishna
3. G Vennela

Department of Computer Science and Engineering, Prasad V. Potluri Siddhartha Institute of Technology (PVPSIT), Vijayawada, Andhra Pradesh, India. Project guide: K. Naga Gopi.

## Evidence

- All 15 paired source outer partitions and 45 corrected calibration-component searches completed.
- Equal-feature L2 mean AUC 0.917447 versus historical selection 0.910316.
- Corrected calibrated/raw component ensembles: mean Brier 0.103731/0.105460; improvement was mixed across repeats.
- Shared-core source mean AUC 0.805884; AngioCAD raw transport AUC 0.712471 under the provisional >50% endpoint.
- Full tests: 116 passed, one optional historical LoRAS-dependency test skipped.
- New experiment verification: 157 listed hash entries matched; historical package checks: 65 artifact plus 74 preservation entries matched.

The report does not claim clinical readiness, prospective prediction, verified independent patients or validation of the historical 38-feature model on AngioCAD. The transported model has a different five-field core, outcome-boundary discrepancy and source/target spectrum differences. Its probability loss was worse than a post hoc source-prevalence constant, which is reported explicitly.

## Files

`CARDIODIAG_RESEARCH_REPORT.pdf` is the final readable report. `CARDIODIAG_RESEARCH_REPORT.md` and `.tex` are editable sources. `figures/` contains standalone vector PDF and PNG assets. `REPORT_NUMERIC_TRACEABILITY.csv` links displayed source/target summaries to saved metric files. `REPORT_RENDER_QA.json` records compilation/page checks; `REPORT_INPUTS.json` records input hashes. `rendered_pages/` contains visual-QA previews.

## Rebuild

From `CARDIODIAG_FRESH`, with `PYTHONPATH=src;.`:

```powershell
.\.venv\Scripts\python.exe paper\research_extension_v1\build_report.py
Push-Location paper\research_extension_v1
& '..\..\..\.tools\tectonic\tectonic.exe' --keep-logs CARDIODIAG_RESEARCH_REPORT.tex
Pop-Location
```

The PDF visual-QA script uses the bundled document runtime with pypdf, pypdfium2 and Pillow. After rebuilding, rerun `qa_report.py` and visually inspect the new pages; old visual-QA approval does not automatically apply to an edited report. The final report is an output artifact, not a historical experiment namespace.

## Remaining human declarations

Corresponding author, contribution statement, funding/conflicts, institutional ethics determination and chosen journal requirements require author/institutional confirmation. No ethics approval or exemption was invented. Completing the computational extension and report does not establish submission acceptance or clinical validation.
