# Reproducibility audit

New checks: eight focused Phase 6/6.1 tests passed in 15.29 seconds using .venv, PYTHONPATH=src;., PYTHONDONTWRITEBYTECODE=1 and pytest cache disabled. The sandboxed interpreter launcher initially could not create its base Python process; the approved outside-sandbox retry succeeded. No full nested experiment or benchmark was launched.

Command from CARDIODIAG_FRESH:

    .\.venv\Scripts\python.exe -m pytest tests\test_phase6_1_audit.py tests\test_phase6_zas_selection.py -q -p no:cacheprovider

MANIFEST_CHECKS.csv independently checks supported direct path-to-SHA256 inputs/outputs mappings in selected manifests. Unsupported nested/list manifest schemas are not silently counted as verified. FILE_INVENTORY.csv inventories project artifacts outside environments/caches. A matching hash proves current byte consistency with a listed hash, not scientific correctness or provenance authenticity.

Historical Phase 7.4.1 reports a 126-package clean install, pip check and four smoke tests; those installation checks were not rerun here. Working requirements are ranged/unpinned; a frozen working-environment snapshot exists separately. Full Ruff, all tests, all notebooks, all manifests and manuscript rendering were not rerun in this pass. No Git worktree exists at the inspected workspace root, so no branch/commit/diff assertion is possible.

The historical 12-page review PDF used ReportLab and a separate local document runtime, not a LaTeX compile or journal template. Its rendering toolchain is not included in the 126-package project snapshot. Fresh document visual verification remains outstanding.
