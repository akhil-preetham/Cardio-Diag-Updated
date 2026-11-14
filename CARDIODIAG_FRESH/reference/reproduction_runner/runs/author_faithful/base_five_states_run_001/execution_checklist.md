# Base Five States run 001 checklist

## Current state

- [x] Dedicated run directory created.
- [x] Run configuration created with execution disabled.
- [x] Prepared working-copy path recorded.
- [x] Notebook and dataset hashes recorded.
- [x] Environment references recorded.
- [x] Output and log directories specified.
- [ ] Explicit first-run approval obtained.
- [ ] Pre-run hashes recomputed.
- [ ] Notebook executed.
- [ ] Outputs independently verified.

## Required pre-run gates

1. Keep `execution_enabled` false until explicit approval.
2. Recompute the source, original-reference, and dataset hashes.
3. Confirm the runner allowlist contains only the prepared working-copy path.
4. Confirm the selected `.venv-repro` interpreter and package versions.
5. Review the side-effect audit and accept or block each operation.
6. Confirm dedicated working, output, and log directories are empty or uniquely versioned.

## Prohibited actions

Do not install packages, modify notebooks, modify datasets, execute other notebooks, silently repair compatibility defects, or report a reproduction result without independent metric verification.
