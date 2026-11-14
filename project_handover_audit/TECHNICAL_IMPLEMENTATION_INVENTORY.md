# Technical inventory

Active research workspace: CARDIODIAG_FRESH. Parent workspace also contains extracted source material, archives and original CSVs. No Git worktree was established (git status reported not a repository). A nested CARDIODIAG_FRESH entry also exists; do not assume it is the active root.

FILE_INVENTORY.csv records paths, sizes and SHA-256 for project files excluding virtual environments and caches. This inventories artifacts; it does not certify every file's semantics.

| Area | Role |
|---|---|
| src/cardiodiag | Data quality/loading, preprocessing pipelines, imbalance sampling, feature selection, ensembles, optimization, evaluation, calibration, explainability, fairness, decision curves and paired statistics |
| runner | Historical reproduction, dataset acquisition/validation, corrected validation, research versions, phased experiments and independent audits |
| runner/phase6_zas_selection/run.py | Groups/grids, fold-local preprocessing, inner choice, selected-score crossfit, sigmoid and threshold, outer prediction, repeat aggregation |
| runner/phase6_zas_selection/finalize.py | Phase 6 finalization; separate from training runner |
| runner/phase6_1_audit/audit.py | Saved-prediction reconciliation and manifest verification |
| tests | Historical unit/structural tests plus Phase 6 and Phase 6.1 focused tests |
| data/raw, data/interim, data/processed, data/metadata | Original acquisitions, validated files, task-specific derivatives and provenance metadata |
| results | Distinct experimental/audit namespaces; names do not alone prove completion |
| reference, notebooks | Author-faithful material and notebook workflows; not executed in this audit |
| paper | Legacy manuscript, research versions, Phase 7 manuscript/corrected copy and later college package |
| requirements.txt, requirements-runner.txt | Ranged/unpinned working requirements |
| results/phase7_4_final_qa/requirements-pinned-working-env.txt | Historical pinned working-environment snapshot; not a general cross-platform lock |
| .venv, .venv-repro, .venv-phase7_4_1 | Different local environments; commands must identify which one is used |

No root pyproject.toml/uv.lock was established from the inspected active-root listing. Do not carry these forward as verified current dependency files merely because the handover mentions them historically.
