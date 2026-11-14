# Clean-environment verification

## Result

**Verified for the frozen project dependency snapshot on Windows x64 / CPython 3.11.16.** The isolated environment was created at `.venv-phase7_4_1`; existing `.venv` and `.venv-repro` were not used as installation targets and were not changed.

The inspected snapshot, `results/phase7_4_final_qa/requirements-pinned-working-env.txt`, documents Windows x64 and CPython 3.11.16. It contains 126 pinned package versions and explicitly excludes the pip/setuptools bootstrap packages. The clean environment reported 126 matching project packages: no missing pins and no extras after excluding those bootstrap packages.

`pip check` reported no broken requirements. Core imports and versions were verified: NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2, pytest 9.2.0, Ruff 0.17.4, SHAP 0.51.0 and imbalanced-learn 0.14.2. Four focused, read-only smoke tests passed (`4 passed`; the suite checks imports, saved manuscript/figure artifacts and Phase 7.4.1/Phase 7.3 hashes). They do not fit models or run benchmarks. The corrected Figure 2 was also regenerated in this environment from saved Phase 6 repeat metrics.

The Phase 7.3 manifest was independently checked after this work: all 46 listed input hashes and all 28 listed output hashes match. This does not provide Git status: this checkout has no `.git` directory or parent Git worktree, so a Git working-tree diff cannot be reported.

## Reproduction commands

Run from the repository root in PowerShell. The first command used the already-present project interpreter to create a separate environment.

```powershell
.\.venv\Scripts\python.exe -m venv .venv-phase7_4_1
.\.venv-phase7_4_1\Scripts\python.exe -m pip install -r results\phase7_4_final_qa\requirements-pinned-working-env.txt
.\.venv-phase7_4_1\Scripts\python.exe -m pip check
.\.venv-phase7_4_1\Scripts\python.exe paper\phase7_4_1_final_manuscript\regenerate_figure2.py
.\.venv-phase7_4_1\Scripts\python.exe results\phase7_4_1_qa\build_manifest.py
.\.venv-phase7_4_1\Scripts\python.exe results\phase7_4_1_qa\verify_manifest.py
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv-phase7_4_1\Scripts\python.exe -m pytest -p no:cacheprovider results\phase7_4_1_qa\test_clean_environment_smoke.py -q
```

The renderer and PDF page-rasterizer used the Codex desktop bundled tool runtime, not the project snapshot: Python 3.12.14, ReportLab 4.4.9, pypdf 6.21.4, pypdfium2 5.13.0 and Pillow 13.0.0. Its current executable and exact render commands are recorded in `MANUSCRIPT_RENDERING_REPORT.md`; that runtime is a local document toolchain, not a portable dependency in the project lock snapshot.

## Limits

The successful install verifies the frozen package closure on this Windows/Python combination. The snapshot is platform-specific, is derived from the former working environment rather than an independently authored lockfile, and omits pip/setuptools versions. It does not establish byte-identical output on another operating system. No training, hyperparameter search, benchmark or historical experiment was run.
