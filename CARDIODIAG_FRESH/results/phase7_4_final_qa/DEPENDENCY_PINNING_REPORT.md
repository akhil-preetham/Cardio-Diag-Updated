# Phase 7.4 — Dependency pinning report

## Snapshot

`requirements-pinned-working-env.txt` was created from `.venv\Scripts\python.exe -m pip freeze --all` on 2026-11-14. It pins the installed direct/transitive dependency closure; only pip and setuptools bootstrap tooling were omitted. This is the inspected working environment, not a clean-install lockfile. `.venv-repro` is separate.

Verified core versions: CPython 3.11.16; NumPy 2.4.6; pandas 3.1.1; SciPy 1.17.1; scikit-learn 1.9.1; Matplotlib 3.11.2; pytest 9.2.0; Ruff 0.17.4. The snapshot also covers notebooks, plotting, explainability, and test packages.

`requirements.txt` is unpinned; `requirements-runner.txt` uses lower bounds only. `environment_baselines/original_venv_current_inventory.txt` cautions that no immutable earlier inventory exists. No package was installed/upgraded and no clean installation was attempted. Resolver/wheel availability, clean imports and byte-identical output generation therefore remain unverified. The snapshot is Windows/CPython/platform specific.

Example disposable install verification (not executed):

```powershell
py -3.11 -m venv .venv-phase7_4-check
.\.venv-phase7_4-check\Scripts\python.exe -m pip install -r results\phase7_4_final_qa\requirements-pinned-working-env.txt
.\.venv-phase7_4-check\Scripts\python.exe -c "import numpy,pandas,scipy,sklearn,matplotlib; print(numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__,matplotlib.__version__)"
```

Do not install into `.venv` or `.venv-repro` for this check.
