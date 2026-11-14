# Phase 6.1 final decision

## Classification

**1. Outer discrimination verified; calibration limitations remain.**

The 1,818 saved predictions match the first 15 frozen outer partitions exactly; all source-row labels and selected-group/model annotations agree. Fold metrics, complete-repeat metrics, and paired repeat differences independently reconstruct to the Phase 6 tables and final decision. No material prediction or discrimination discrepancy was found.

Calibration has a real limitation: selected candidate/hyperparameters are chosen using all outer-training labels before inner OOF scores are used to fit the sigmoid and choose threshold. The crossfit scores are therefore estimator-out-of-fold but selection-dependent. Inner-fit models use about two thirds of the outer-training rows while the final predictor is refit on all of them. The calibration selection metadata also has four labels for a three-fold loop; this is a reporting-only defect, corrected in the new audit CSV.

This limitation does not expose outer-test labels to prediction selection or calibration. The outer test labels are used for evaluation and post-fit permutation importance only. Accordingly, outer discrimination remains verified as the performance of the complete training procedure; calibration evidence should be described as limited and must not be presented as independent calibration validation for a fixed final model.

## Supported, limited and unresolved

- **Supported:** source-row prediction coverage, label matching, per-fold and per-repeat metric reconstruction, paired comparison arithmetic, Phase 6 manifest file hashes, and the 4-versus-3 metadata error.
- **Limited:** calibration construction and calibration/threshold estimates; only three repeated outer splits; rows are not verified patients; calibration slope/intercept on small fold samples.
- **Unresolved:** patient identity/independence, feature acquisition timing, external generalization, and whether calibration would transport to a fixed model trained on all rows.

No patient-level confidence claim or external-validation claim is made. Historical Phase 1–6 files were not changed.

## Reproduction commands

From the project root in PowerShell, the documented project virtual environment is `.venv\Scripts\python.exe` (verified to resolve to the project-local interpreter). Run only the read-only audit and focused tests:

```powershell
$env:PYTHONPATH = 'src;.'
.\.venv\Scripts\python.exe -m runner.phase6_1_audit.audit
.\.venv\Scripts\python.exe -m pytest tests\test_phase6_1_audit.py -q
```

Do not run `runner.phase6_zas_selection.run` to reproduce this audit; that command refits the Phase 6 candidate pool and is unnecessary here. The audit reads saved predictions/splits, recalculates metrics, verifies hashes and writes only `results/phase6_1_audit/`.
