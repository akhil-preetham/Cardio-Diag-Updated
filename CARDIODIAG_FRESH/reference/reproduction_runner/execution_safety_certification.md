# Execution safety certification

## Static code findings

### Enforced controls

- `runner/execute_notebooks.py` exits before importing `nbclient` when `notebook_execution.enabled` is false.
- If enabled, an empty allowlist is rejected.
- Selection is exact relative-path allowlisting; there is no wildcard expansion.
- Notebooks are copied to a working directory before execution.
- Executed outputs are written to a separate results directory.
- The loop is sequential.
- Exceptions are captured in the manifest and traceback file; source notebooks are not overwritten.
- The runner does not install packages or invoke package managers.
- `run_reproduction.ps1` uses explicit modes and checks subprocess exit codes.

### Important limitations

- The configured allowlist points to the extracted author notebook, not the prepared local-path working copy. Because execution is disabled this is dormant, but it must be corrected and explicitly approved before any author run.
- The skip patterns are advisory; the exact allowlist remains authoritative.
- The runner does not itself enforce a historical environment, dataset hash, or compatibility decision.
- Notebook-internal shell, network, deletion, or package-install operations are not automatically sandboxed by the runner. Static review is required before approval.
- `corrected_validation.enabled` is true in the configuration, while notebook execution is false. The PowerShell `corrected` mode can therefore run model training if explicitly invoked; it was not invoked here.

## PowerShell safety

`run_reproduction.ps1` defaults to `all-safe`, whose documented path performs preflight/audit/dataset validation and does not call the notebook runner. The `author` and `corrected` modes are explicit, but the script does not add a separate interactive approval gate.

## Certification result

Notebook execution is disabled and source protection/output isolation are implemented. The system is not certified for execution yet because the allowlist target mismatch, notebook-internal operations, and missing final approval remain unresolved.
