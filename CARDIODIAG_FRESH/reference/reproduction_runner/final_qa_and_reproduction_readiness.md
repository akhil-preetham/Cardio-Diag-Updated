# Final QA and reproduction readiness

## Verified checks

- Original notebook SHA-256 matches the recorded authoritative value: `990f24622ca3aefe2f62e37f7d9e1097fe14b118a49ee398ff3ee67568bfbe4d`.
- Prepared notebook SHA-256 matches the patch manifest and `SHA256SUMS.txt`: `666ab88c049dba064057e79d55fb413bca37a1c7de3f3bb8c24cad070844b646`.
- The author repository is clean.
- The source inventory and readiness matrix both contain 18 unique notebook paths, with identical path sets.
- All 18 referenced extracted notebooks exist.
- The experiment template parses and defaults execution to false.
- `config/runner_config.json` has `notebook_execution.enabled=false`.
- `pip check` passed for the recorded `.venv-repro` environment.
- No notebook, kernel, model, or GridSearchCV execution was invoked during this QA stage.

## Failed checks

- The previous aggregate verification shell returned exit code 1 because its final `Get-Process python,jupyter` probe returned no matching process and was not normalized as a successful negative observation.
- The previous environment record contains a working-copy hash transcription error. A corrected, separately named record was created; the original was preserved.

## Incomplete or unverified checks

- No supplied package record proves that a historical environment supports the complete original notebook.
- No prior package inventory for `.venv` is available, so unchanged status of that environment is unverified.
- The current `.venv-repro` package inventory matches the available dependency record, but historical immutability cannot be proven from a single snapshot.
- The configured author allowlist targets the extracted notebook rather than the prepared working copy; this requires correction before approval.
- Notebook-internal shell, network, deletion, and package-install behavior has been statically flagged but not dynamically tested.

## Remaining blockers

Invalid/legacy grid expressions, the KNN spelling error, the XGBRegressor/classification scorer mismatch, unavailable optional dependencies and external files, unresolved historical versions, and unresolved allowlist targeting remain.

## Required approvals before execution

1. Approve Track A as either a historical-environment attempt or an intentionally recorded faithful failure.
2. Resolve the allowlist to the exact approved working-copy path.
3. Confirm source, dataset, environment, and output hashes.
4. Review notebook-internal external/destructive operations.
5. Approve one sequential run with dedicated outputs.

## Final assessment

The documentation and matrix QA are substantially complete, but execution authorization is not justified. No result may be described as reproduced.

`NOT READY — REQUIRED EVIDENCE OR SAFETY WORK REMAINS`
