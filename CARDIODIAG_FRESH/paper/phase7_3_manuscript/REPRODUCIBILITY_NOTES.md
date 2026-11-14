# Reproducibility notes

The manuscript was assembled from immutable saved artifacts; no model training or candidate search was run for Phase 7.3. The saved model experiment and independent audits can be reproduced from a clean copy of the repository. Do not run commands in the current evidence checkout if preserving its completed output namespaces is required.

## Phase 6 ZAS selection-aware experiment

From a clean project copy with the project environment installed, PowerShell:

```powershell
$env:PYTHONPATH = 'src;.'
.\.venv\Scripts\python.exe -m runner.phase6_zas_selection.run
.\.venv\Scripts\python.exe -m pytest tests\test_phase6_zas_selection.py -q
.\.venv\Scripts\python.exe -m runner.phase6_zas_selection.finalize
```

This refits the 30-candidate selection procedure across 15 outer partitions and writes Phase 6 outputs. The runner requires an empty output namespace except the frozen protocol. It was not run while preparing this paper.

## Phase 6.1 read-only saved-result audit

```powershell
$env:PYTHONPATH = 'src;.'
.\.venv\Scripts\python.exe -m runner.phase6_1_audit.audit
.\.venv\Scripts\python.exe -m pytest tests\test_phase6_1_audit.py -q
.\.venv\Scripts\python.exe -m ruff check runner\phase6_1_audit tests\test_phase6_1_audit.py
```

The audit checks the existing predictions and outputs; it does not rerun candidate selection. See `results/phase6_1_audit/PHASE6_1_FINAL_DECISION.md` for its verified command/environment description.

## UCI four-centre analysis

Original experiment commands are documented in the Phase 4 protocol and reports. Phase 4.3 reconstruction/audit uses `.venv-repro` and its documented runner. These commands regenerate existing output namespaces and were not run during Phase 7.3. The manuscript values are traced to Phase 4.3 saved metric/prediction artifacts and independent reconciliation.

## Figure and table regeneration

From the project root:

```powershell
.\.venv\Scripts\python.exe paper\phase7_3_manuscript\build_tables_figures.py
```

This script only reads saved CSV artifacts and writes tables/figures inside `paper/phase7_3_manuscript/`. Environment versions in the Phase 6.1 audit include Python 3.11.16, NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1 and scikit-learn 1.9.1. Matplotlib version in the project-local venv was 3.11.2. LaTeX compilation remains pending because no `pdflatex`, `xelatex`, `lualatex` or `latexmk` executable was available on the project PATH.
