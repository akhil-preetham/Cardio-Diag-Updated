# Complete execution workflow

## Status

This is a static workflow design. Notebook execution, kernel startup, model training, package installation, and environment changes are prohibited for this stage.

## Gates

1. Confirm the original notebook hash and prepared-copy hash.
2. Confirm `reference/author_release` is clean.
3. Confirm `config/runner_config.json` has `notebook_execution.enabled=false`.
4. Confirm the selected track has a dedicated working copy and output root.
5. Confirm dataset hashes, schema, row count, target, and required external files.
6. Record Python and package versions without changing the environment.
7. Obtain explicit execution approval for exactly one allowlisted notebook or track.

## Sequential execution policy

Only one model-heavy notebook may run at a time. The runner must copy the approved notebook to a unique run directory, never execute the source artifact, and write outputs outside the source tree. It must capture stdout, stderr, warnings, exceptions, traceback, runtime, cell status, package versions, dataset hashes, and the final run status.

Failures are terminal evidence for that run. Partial execution must be recorded as partial and must not be reported as completion. A failed run must not trigger automatic source edits, retries with changed parameters, or package changes.

## Suggested order

1. Track A base notebook, only after a historical-environment decision.
2. Track B base notebook, only from a new copy if Track A is blocked by implementation compatibility.
3. Track C corrected validation independently.
4. Full-methodology, fairness, XAIH-DRC, LIME, statistical, and clinical-utility notebooks only after their inputs and package requirements are individually approved.

## Recovery

Each run receives a unique identifier and a manifest. Completed notebooks are skipped only when the manifest proves the same source hash, dataset hashes, environment fingerprint, and track. Incomplete runs remain inspectable and are never silently resumed with different software or data.
