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
.\.venv-phase7_4_1\Scripts\python.exe paper\phase7_4_1_final_manuscript\regenerate_figure2.py
```

The Phase 7.4.1 figure-only script reads saved Phase 6 repeat metrics and writes only Figure 2 in this versioned directory. It uses an explicit 0–1 ROC-AUC scale and 300-dpi PNG/PDF output. The Phase 7.3 full builder is retained in its original package and must not be run against this corrected copy. The pinned project dependencies were verified in `.venv-phase7_4_1` (Python 3.11.16; NumPy 2.4.6; pandas 3.1.1; SciPy 1.17.1; scikit-learn 1.9.1; Matplotlib 3.11.2). No LaTeX compiler is installed; the manuscript PDF is rendered from Markdown with the local ReportLab document renderer, not compiled from LaTeX.

## PDF review rendering

The Phase 7.4.1 review PDF and page verification are reproduced from the repository root using the current Codex desktop bundled runtime (Python 3.12.14; ReportLab 4.4.9; pypdf 6.21.4; pypdfium2 5.13.0; Pillow 13.0.0):

```powershell
& "C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe" results\phase7_4_1_qa\render_markdown_manuscript.py
& "C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe" results\phase7_4_1_qa\verify_render.py
```

The bundled-runtime path is specific to this desktop installation; it is not part of the pinned project environment. The rendering report records this limitation and the inspected output.
