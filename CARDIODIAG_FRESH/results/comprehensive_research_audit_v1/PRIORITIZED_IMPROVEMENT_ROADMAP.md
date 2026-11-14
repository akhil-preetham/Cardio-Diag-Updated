# Prioritized improvement roadmap

## Phase 1 — provenance and claims (low cost, blocking)

1. Reconcile paired-statistics documentation and artifacts.
2. Create the authoritative experiment registry proposed in `NEXT_ACTIONS.md`.
3. Update manuscript tables so every value resolves to one saved output.
4. Keep author-faithful, compatibility-adapted, and corrected labels separate.

## Phase 2 — leakage-resistant evaluation (moderate cost, blocking)

1. Prespecify nested or repeated grouped validation.
2. Persist row IDs for every train/test prediction vector.
3. Integrate fold-local feature selection and any LIME-derived input into the training pipeline.
4. Add independent tests for split isolation and paired-row alignment.

## Phase 3 — external validity (high cost, blocking for clinical claims)

1. Obtain and hash the missing benchmark dataset.
2. Obtain an independent cardiovascular cohort with compatible target definition.
3. Evaluate transportability and external calibration without mixing datasets.

## Phase 4 — robustness and clinical interpretation (moderate/high cost)

1. Add calibration uncertainty and prespecified threshold analysis.
2. Report subgroup counts and uncertainty with multiplicity-aware interpretation.
3. Validate decision-curve assumptions and avoid clinical deployment language.

## Phase 5 — algorithmic contribution (only after Phases 1–3)

Compare source-shaped OVEC/OSEC and a prespecified corrected baseline under the same observations and protocol. Do not launch broad optimization until the evaluation boundary and registry are stable.

## Dependencies and acceptance

Phase 2 depends on Phase 1. Phase 3 depends on compatible data and a frozen protocol. Acceptance requires reproducible manifests, no untracked test-set selection, aligned prediction IDs, and manuscript/artifact agreement.
