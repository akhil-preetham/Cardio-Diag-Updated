# Phase 7.4 — Reproducibility audit

## Environment and execution instructions

The project-local `.venv\Scripts\python.exe` is inside the repository and reports CPython 3.11.16 on Windows AMD64. Directly confirmed versions: NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2, pytest 9.2.0 and Ruff 0.17.4. `.venv-repro` is a separate package environment, also Python 3.11.16. `requirements-pinned-working-env.txt` captures the full `.venv` dependency closure (excluding pip/setuptools); it is a provenance snapshot, not a portable lock.

Root `requirements.txt` uses bare package names; `requirements-runner.txt` uses open minimum bounds. Neither freezes the inspected environment. `run_reproduction.ps1` selects `.venv-repro` when present, otherwise PATH Python; modes can write/replace outputs. It sets `PYTHONPATH=runner`, while Phase 6 module commands use `src;.`. Keep these tracks distinct. Phase 4.3 uses `.venv-repro`; Phase 6/6.1 uses `.venv`.

Phase 7.3 reproduction notes give the Phase 6 interpreter but no environment-creation command. Phase 4.3 and Phase 6.1 commands are in their reports, but regenerate existing output namespaces. Run only in a disposable copy after reviewing their effects. Patient identity and acquisition timestamps are not available; the evaluation unit remains source rows. Outcomes and task definitions were not changed in this audit.

## Verification status

| Item | Status |
|---|---|
| Environment inspection | Completed; interpreter and installed versions checked. |
| Dependency pin snapshot | Completed; full installed `.venv` closure recorded. |
| Clean environment install | Not performed; no packages installed or upgraded. |
| Unit tests | Not run; Phase 6.1 test fixture calls reconciliation that writes into the preserved historical audit namespace. |
| Read-only QA | Phase 7.3: 74, Phase 7.1: 25, Phase 7.2: 41 manifest entries rehashed; zero mismatches. |
| End-to-end reproduction | Not performed; no model fitting, search, audit runner or benchmark rerun. |
| PDF QA | PDF files exist. PNGs are 2301–3039 px wide at 300 dpi. PDF raster/pixel equivalence not checked; no TeX compilation claimed. |

## Exact reproduction commands (not executed)

From repository root in PowerShell. These commands may rewrite their output namespaces; use a clean copy.

```powershell
$env:PYTHONPATH = 'src;.'
.\.venv\Scripts\python.exe -m runner.phase6_1_audit.audit
.\.venv\Scripts\python.exe -m pytest tests\test_phase6_1_audit.py -q
.\.venv\Scripts\python.exe -m ruff check runner\phase6_1_audit tests\test_phase6_1_audit.py

$env:PYTHONPATH = 'src;.'
.\.venv-repro\Scripts\python.exe runner\phase4_validation_audit\run.py
.\.venv-repro\Scripts\python.exe runner\phase4_validation_audit\make_figures.py
.\.venv-repro\Scripts\python.exe -m pytest -q
.\.venv-repro\Scripts\python.exe -m ruff check runner\phase4_validation_audit tests\test_phase4_validation_audit.py

.\.venv\Scripts\python.exe paper\phase7_3_manuscript\build_tables_figures.py
```

The last command rewrites frozen Phase 7.3 outputs and is shown only to identify the build entry point; do not run against the preserved package. Phase 6 model-fitting commands are intentionally omitted from this audit.
