# Final first-run approval certification

## Verified hashes

- Original notebook: `990f24622ca3aefe2f62e37f7d9e1097fe14b118a49ee398ff3ee67568bfbe4d`
- Prepared notebook: `666ab88c049dba064057e79d55fb413bca37a1c7de3f3bb8c24cad070844b646`
- Dataset: `e4068802a044e8a1e3ae9fa81deae7025711bf4f13e3057f96e4f1cf3b0180c2`

All agree with the patch manifest, `SHA256SUMS.txt`, corrected environment record, and first-run input manifest.

## Runner safeguards

- Allowlist contains only the prepared working copy.
- Extracted original notebook is not an allowlist target.
- Execution remains disabled.
- Explicit allowlisting and `approval_granted=true` are required before execution.
- Source is copied before execution.
- Source hashes are checked before and after execution.
- Existing working-copy and output artifacts are not overwritten.
- Execution is sequential with timeout and exception capture.
- No package installation is performed by the runner.
- Failure cannot write the source notebook.

Static syntax checks passed for the modified Python runner. The runner was not launched.

## Environment records

The current `.venv` baseline is recorded at `environment_baselines/original_venv_current_inventory.txt`. No earlier immutable `.venv` inventory exists, so historical equivalence cannot be proven. The recorded `.venv-repro` inventory remains unchanged for this task.

## Run-package completeness

The five required files exist and refer to the same prepared notebook and dataset hashes:

- `run_config.json`
- `input_hash_manifest.json`
- `environment_record_reference.txt`
- `execution_checklist.md`
- `methodology_record.md`

The run configuration has `execution_enabled: false`.

## Outstanding limitations

The author notebook still contains documented compatibility and methodological defects. A first run may fail in a controlled manner; such a failure must not be silently repaired or reported as successful reproduction. Notebook-internal side effects and package-version compatibility require explicit pre-run acceptance.

## Exact command for a future approved run

The command is intentionally not executed here:

```powershell
.\run_reproduction.ps1 -Mode author
```

Execution requires explicit user approval after rechecking hashes, environment versions, side effects, and the disabled configuration.
