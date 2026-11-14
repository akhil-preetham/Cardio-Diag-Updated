# Three-track execution design

## Restrictions

This design is static only. No notebook, kernel, model, or package operation was performed.

## Track A — `author_faithful`

**Purpose:** determine what the preserved author implementation does under the closest defensible environment.

**Source policy:**

- Use the prepared working copy only.
- Preserve cells 5 and 9–20 exactly.
- Permit only the existing cell 2 Colab-mount omission and cell 4 local dataset path.
- Do not remove invalid grid candidates or change estimator/metric pairings.
- If execution fails, preserve the traceback and classify the failure; do not repair it silently.

**Required evidence:**

- Original and working-copy hashes immediately before execution.
- Exact package versions and Python version.
- Dataset hash and schema.
- One sequential execution manifest entry.
- Per-cell traceback and duration.
- Explicit distinction between saved notebook outputs and newly generated outputs.

**Expected outcome:** likely a controlled failure at an invalid grid/API operation unless a verified historical environment is found. A failure is still evidence about reproducibility.

## Track B — `compatibility_adapted`

**Purpose:** assess whether minimal implementation/API adaptations allow the source workflow to run.

**Directory:** `reference/reproduction_runner/compatibility_adapted/`

**Source policy:**

- Start from a fresh copy of the prepared working copy.
- Never modify Track A.
- Record every changed cell in a separate patch manifest.
- Candidate changes include removing `max_features='auto'`, correcting `minkowskii`, filtering invalid logistic solver/penalty combinations, and translating legacy XGBoost parameters.
- Do not change SMOTE placement, split ratio, test-set stopping, or estimator/metric meaning under this track unless separately promoted to Track C.

**Labels:** all outputs must say `compatibility_adapted`; they cannot be described as exact author-faithful results.

**Required evidence:**

- Before/after source hashes.
- Complete cell-level patch manifest.
- Rationale and expected behavior for each API change.
- Separate output and result tables.
- Package/API evidence for each changed parameter.

## Track C — `corrected_validation`

**Purpose:** answer the leakage-controlled scientific question, not reproduce the original implementation.

**Directory:** `reference/reproduction_runner/corrected_validation/`

**Source policy:**

- Use an independent implementation, not a modified author notebook.
- Preprocessing fitted only within training folds.
- Stratified 75:25 split where following the paper comparator.
- SMOTE only inside training folds.
- Nested model selection or a strictly untouched test set.
- No test-set-driven stopping or threshold selection.
- Appropriate classifier/scoring pairing.
- Persist row identifiers, predictions, metrics, seeds, and package versions.

**Labels:** all outputs must say `corrected_validation`; they cannot be merged with Tracks A or B.

## Execution order and gates

1. Verify hashes and repository cleanliness.
2. Approve an exact Track A allowlist or explicitly approve a controlled faithful failure.
3. If Track A fails for compatibility reasons, decide whether to prepare Track B.
4. Run Track C independently; never use Track A/B outputs as hidden inputs.
5. Generate separate reports and result tables.

## Recommendation

Track A should be attempted first only as a controlled, failure-tolerant reproduction attempt after a historical environment decision. If no historical environment is approved, the safest next action is to document the faithful failure and then prepare Track B separately. Track C may proceed independently but must not be presented as author reproduction.
