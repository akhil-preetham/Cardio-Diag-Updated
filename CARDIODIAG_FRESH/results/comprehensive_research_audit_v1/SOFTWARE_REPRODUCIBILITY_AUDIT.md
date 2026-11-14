# Software and reproducibility audit

## Strengths

- `.venv-repro` is kept separate from `.venv`.
- Notebook execution is disabled behind an approval/allowlist gate in `config/runner_config.json`.
- Dataset, notebook, model, and output hashes are recorded in existing manifests.
- Controlled runners use fixed seeds and isolated output directories.
- Saved models are reloaded and prediction equality is checked in the final verification.
- Focused project tests exist, and prior verification reports eight tests, Ruff, and compilation passing for the relevant scope.

## Risks

| Severity | Finding | Evidence |
|---|---|---|
| High | No historical lockfile/equivalence proof | `requirements-runner.txt`, `reference/reproduction_runner/audit/dependency_audit.json` |
| High | Contradictory status of paired statistics | `docs/EXPERIMENT_RESULTS.md` vs `docs/FINAL_EXECUTION_VERIFICATION.md` |
| Medium | Full project Ruff has unrelated pre-existing findings | `results/final_independent_verification.json` |
| Medium | Registry is distributed across reports/manifests | `results/`, `docs/`, `reference/reproduction_runner/` |
| Low | Notebook execution records and controlled Python runs use different provenance schemas | existing manifests |

## Preservation result

This audit writes only to `results/comprehensive_research_audit_v1/`. Original CSVs, extracted author notebooks, and existing experiment outputs are not modified or deleted. Source and output hashes are recorded in `audit_manifest.json`.

## Recommended engineering correction

Introduce a single immutable experiment registry with source hashes, configuration hashes, prediction row IDs, model hashes, protocol labels, and status. Add tests that reject paired comparisons without matching row IDs and reject reports whose metric source cannot be resolved.
