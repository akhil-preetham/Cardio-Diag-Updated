# Runner safety remediation report

## Changes made

1. The dormant allowlist now contains only:

   `reference\reproduction_runner\work\author_faithful\base_reproduction\Daza_Paper_Exact_Accuracy.ipynb`

2. `runner/execute_notebooks.py` now resolves explicit allowlist entries from the project root instead of discovering and iterating through every extracted notebook.
3. Each selected source notebook is hashed before copying and checked again after the run. A changed source hash prevents a success status.
4. The configured dataset is hashed before execution and a mismatch fails closed.
5. Missing or non-notebook allowlist targets fail closed.
6. Existing working-copy or output artifacts cause a fail-closed refusal instead of overwrite.
7. An explicit `approval_granted` flag is required in addition to `enabled=true`.

Notebook execution remains disabled.

## Static runner audit

- Copy before execution: enforced with `shutil.copy2`.
- Source overwrite protection: source is never used as the output path.
- Hash checks: added before and after execution.
- Dedicated outputs: `reference/reproduction_runner/author_faithful/executed_notebooks`.
- Sequential execution: one target is processed at a time.
- Timeout: passed to `NotebookClient`.
- Error handling: exception, traceback, event log, and manifest entry are preserved.
- Environment selection: `run_reproduction.ps1` selects `.venv-repro` when present; exact package versions are recorded separately.
- Package installation: no installation code exists in the runner.
- Automatic execution: no execution occurs on import; `main` runs only under the command-line guard.
- Allowlist expansion: no recursive notebook discovery remains in the execution path.
- Failure isolation: failures are recorded and do not write to the source notebook.

## Remaining limitation

The runner does not itself enforce the package-version manifest or inspect notebook-internal side effects. These remain pre-run approval gates. The corrected configuration is still disabled and has not been passed to the runner.
