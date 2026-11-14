# Documentation inventory and recovery audit

## Repository evidence

`CARDIODIAG_FRESH` has no Git metadata. The nested `reference/author_release` repository is clean at commit `e63b02c8a2aba6344a48025f657696fd99ac3a5f` and tracks only the author release README, archives, datasets, and `Datasets`; it does not contain project documentation.

Git therefore cannot restore project documentation without risking unrelated files. No documentation was restored from Git.

## Current documentation

The current project documentation is:

- `README.md`
- `docs/DATASET_DOCUMENTATION.md`
- `docs/ENVIRONMENT_SETUP.md`
- `docs/EXPERIMENT_RESULTS.md`
- `docs/FINAL_COMPLETION_REPORT.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/METHODOLOGY.md`
- `docs/PROJECT_COMPLETION_STATUS.md`
- `docs/REPRODUCIBILITY_AUDIT.md`
- `docs/REPRODUCTION_GUIDE.md`
- `docs/SOURCE_TO_PAPER_MAPPING.md`
- this inventory

## Prior destructive operation

The preceding workflow history records an accidental PowerShell deletion of files in `docs`. The affected documentation was recreated as separately authored project documentation. Because the workspace is not Git-tracked, original byte-for-byte contents and any additional untracked files cannot be recovered or proven from repository history.

Known recreated deliverables include the final completion, reproducibility audit, source mapping, limitations, methodology, environment, dataset, reproduction guide, and experiment-results documents. No research outputs, original notebooks, author datasets, or completed JSON reports were deleted or overwritten by this recovery work.

## Missing documentation

No manuscript source existed in `paper/` before this completion pass. A new validated manuscript source is supplied at `paper/manuscript.tex`; it does not claim authorship, affiliation, funding, ethics approval, or publication status.

The exact historical author documentation and the requested paper table extraction remain unavailable.
