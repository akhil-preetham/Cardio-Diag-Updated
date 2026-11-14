# Model and experiment audit

## Executed performance

| Track/result | Five States ROC-AUC | Mendeley ROC-AUC | Interpretation |
|---|---:|---:|---|
| Author-faithful prepared notebook | 0.9555 | not available | Single source-shaped run with documented leakage risks |
| Corrected repeated validation | 0.9332 mean | 0.9964 mean | Ten overlapping stratified holdouts; not external validation |
| Corrected OSEC fixed holdout | 0.9424 | 0.9995 | Single controlled comparison holdout |
| Corrected OVEC fixed holdout | 0.9421 | 0.9982 | Single controlled comparison holdout |

The authoritative metric files are `results/experiment_metrics.json`, corrected validation reports, and `results/ensemble_experiments/**/metrics.json`. The two-track distinction must be retained.

## OVEC/OSEC

The source-shaped OVEC has KNN, random forest, logistic regression, and SVM components with weighted hard/soft voting. The source-faithful search includes invalid `n_estimators=None`, so the faithful run fails. The compatibility rerun uses finite `n_estimators=2`; it is not exact reproduction. OSEC uses RF, tree, gradient boosting, Gaussian NB, and a logistic meta-learner with five-fold stacking.

## Statistical validity

The paired comparison CSV contains matched-looking metrics and Holm-adjusted Wilcoxon values, but the documentation contradiction must be resolved. Any paired test is valid only when both prediction vectors use identical observations and ordering. The existing code checks vector lengths, not semantic row IDs; manifests should record row indices.

## Calibration, robustness, and generalization

Calibration and decision-curve artifacts are present, but internal repeated splits do not establish transportability. Cross-dataset evaluation should not be forced unless feature names, encoding, target definitions, and sampling frames are compatible. No independent external cohort is present.

## Primary metrics

For imbalanced cardiovascular classification, prespecify ROC-AUC and PR-AUC as ranking metrics, balanced accuracy/sensitivity/specificity/F1 as operating-point metrics, and Brier/calibration summaries for probabilities. Accuracy alone is inadequate.
