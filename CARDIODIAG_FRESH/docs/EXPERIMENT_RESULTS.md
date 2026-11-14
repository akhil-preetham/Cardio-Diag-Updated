# Experiment results

All values below are extracted from saved outputs. Machine-readable records are in `results/experiment_metrics.json`, `results/research_results_summary.csv`, and `results/dataset_comparison.csv`.

## Author-faithful track

The prepared working copy executed successfully for 404 seconds. The earlier attempt against the unadapted copy failed at `from google.colab import drive`; that failure is preserved in the execution manifest and is not treated as the reproduction result.

| Dataset | Accuracy | Precision | Recall/sensitivity | F1 | ROC-AUC | Balanced accuracy | Specificity | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Five States | 0.8922 | 0.9052 | 0.9052 | 0.9052 | 0.9555 | not recorded | not recorded | not recorded | not recorded |

The notebook used SMOTE before the 80:20 split, `random_state=42`, five-fold accuracy GridSearchCV, calibrated stacking, and test-set-driven stopping at epoch 6. The saved output contains non-finite search scores, invalid/legacy candidates, LogisticRegression and SVC deprecation/convergence warnings, and an XGBRegressor/classification-accuracy mismatch. Selected parameter strings are preserved in `experiment_metrics.json`.

The executed notebook and manifest agree on the reported metrics and source hash:

- Executed output: `reference/reproduction_runner/author_faithful/executed_notebooks/Daza_Paper_Exact_Accuracy_a788cef8a6_executed.ipynb`
- Prepared source SHA-256 before/after: `666ab88c049dba064057e79d55fb413bca37a1c7de3f3bb8c24cad070844b646`
- Five States dataset SHA-256: `e4068802a044e8a1e3ae9fa81deae7025711bf4f13e3057f96e4f1cf3b0180c2`

## Corrected-validation track

The corrected runner used ten seeded repeated stratified 75:25 holdouts, five-fold stratified CV within each training partition, fold-local preprocessing, training-fold SMOTE, ROC-AUC selection, and one test evaluation after selection.

| Dataset | Accuracy mean (SD) | Balanced accuracy mean (SD) | Precision mean (SD) | Sensitivity mean (SD) | Specificity mean (SD) | F1 mean (SD) | ROC-AUC mean (SD) | PR-AUC mean (SD) | Brier mean (SD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Five States | 0.8783 (0.0160) | 0.8755 (0.0158) | 0.8809 (0.0154) | 0.9016 (0.0230) | 0.8495 (0.0216) | 0.8910 (0.0148) | 0.9332 (0.0088) | 0.9322 (0.0113) | 0.0954 (0.0078) |
| Mendeley, `patientid` excluded | 0.9772 (0.0076) | 0.9759 (0.0079) | 0.9768 (0.0096) | 0.9841 (0.0098) | 0.9676 (0.0136) | 0.9804 (0.0065) | 0.9964 (0.0028) | 0.9970 (0.0026) | 0.0182 (0.0058) |

Each corrected run had 688/230 Five States train/test observations and 750/250 Mendeley train/test observations. The final estimator grid was `C in {0.1, 1.0, 10.0}` and the selected estimator was a stacking classifier with RF, decision tree, gradient boosting, Gaussian NB, and logistic-regression final estimator.

## Statistical and explainability analyses

- Conditional bootstrap intervals for accuracy and ROC-AUC were computed per saved test-prediction run in `results/results_bootstrap_confidence_intervals.csv`. They are not independent-cohort confidence intervals because repeated holdouts overlap.
- Descriptive Sex subgroup metrics with `n >= 20` are in `results/subgroup_performance.csv`. They are not a fairness certification.
- Ten-bin ECE/MCE summaries are in `results/calibration_summary.csv`.
- DeLong, Wilcoxon, and paired model comparisons were not performed: no competing models have persisted predictions on identical observations.
- Existing LIME artifacts were not relabelled as newly reproduced; their provenance and model-input alignment were not independently established.

## Interpretation

The corrected results are a separate comparator, not an author-faithful reproduction and not independent external validation. No clinical deployment claim follows from either track.
