# Phase A — Historical integrity report

## Preserved evidence

No historical result file, original dataset, notebook, or phase manifest was modified during inspection. The current completion artifacts are new and isolated under `results/final_project/`. The project and its parent have no `.git` directory, so tracked/untracked status and a clean working tree cannot be verified.

The Phase 5 manifest contains 40 output entries; all 40 files existed and matched their listed hashes at inspection. Phase 4 and Phase 4.3 also have reproducibility manifests, but this report does not claim that every manifest in every earlier namespace was fully rehashed here.

## Corrections to interpretation

1. ZAS metrics are reproduced from persisted predictions and the 303 unit is a source row. Patient-level identity is unavailable.
2. Logistic and Ridge are duplicate model labels and have identical predictions.
3. The 0.928 candidate was not selected in a nested feature/model search and remains exploratory.
4. The Phase 4 duplicate-sensitivity subset is invalid because exclusions omit centre identity and the recorded exclusion count is calculated after filtering. The underlying Phase 4 pooled metrics and centre-keyed paired comparisons are separate, but no robustness conclusion should rely on the saved duplicate-sensitivity rows.
5. The paired-statistics statement in the earlier Five States/Mendeley results document refers to the ten-run comparator; later paired tests are for fixed-holdout OVEC/OSEC. Scope must be explicit.
6. Phase 5 UCI records `validation="stratified_inner"` in prediction rows even though `run.py` fits a fixed C=1 logistic model on three centres and predicts the held-out centre without inner CV. Read the code and centre key as authority; the stored validation label is inaccurate.
7. Phase 5.1's output code writes fixed filenames and is not safe to rerun if historical artifacts must remain immutable.

## Historical authority

Use the frozen V3 protocol/configuration and its prediction artifacts as the ZAS reference; use the Phase 4.3 source prediction table and corrected audit for UCI staged comparisons. Phase 5 provides descriptive exploratory candidates and separate fixed-parameter comparisons, not a replacement for these authorities. External validation remains unavailable for the frozen ZAS model because no local candidate is demonstrated to have matching endpoint, all required predictors, verified acquisition timing and independence.
