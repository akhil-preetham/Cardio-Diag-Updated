# Phase 7.4 — Correction log

## Manuscript corrections

No numerical or scientific interpretation error requiring a new manuscript version was found. The Phase 7.3 manuscript, tables, figures and manifest remain unchanged; no `paper/phase7_4_manuscript/` copy was needed.

## Audit findings and actions

| Item | Action/status | Reason |
|---|---|---|
| Dependency pin | Added `requirements-pinned-working-env.txt` only in this audit namespace. | Existing requirements are unpinned/open-range; snapshot records inspected environment without modifying historical files. Clean installation remains unverified. |
| Reproduction notes | Documented actual `.venv` / `.venv-repro` use and write effects. | Avoid environment ambiguity and accidental overwriting. |
| Figure 2 scale | Recommendation only; figure not edited. | Truncated axis can emphasize the difference; numeric values are accurate and visible. |
| Tests | Not executed. | Existing Phase 6.1 fixture writes into its preserved audit namespace. No tests-passed claim is made. |
| Clean install/PDF render | Not performed. | Remains an explicit technical/production verification task. |
