# Decision on the ZAS ROC-AUC 0.928 result

## Classification: **EXPLORATORY ONLY**

The result is not classified as invalid arithmetic or proven direct leakage. The implementation excludes the obvious target-derived and downstream fields, and the candidate-level pipeline uses fold-local preprocessing and inner hyperparameter selection. However, it cannot support a defensible prediagnostic CAD claim because:

1. 36 of 38 included variables are only clinically plausible, not source-documented as available before diagnostic confirmation.
2. The feature group/model row was selected from 36 benchmark candidates after observing performance; feature-group choice was not nested.
3. The cohort has no patient identifier and the source timing, units and ascertainment definitions are incomplete.
4. The reported confidence is based on repeated-CV prediction aggregation, not an untouched independent cohort.

## Traced benchmark rows

```text
    feature_group    model   n  events  roc_auc  pr_auc  sensitivity  specificity  precision      npv       f1  balanced_accuracy    brier  log_loss  calibration_intercept  calibration_slope
eligible_combined logistic 303     216 0.928214 0.96683     0.925926     0.712644   0.888889 0.794872 0.907029           0.819285 0.096424  0.314108              -0.167191           1.307457
eligible_combined    ridge 303     216 0.928214 0.96683     0.925926     0.712644   0.888889 0.794872 0.907029           0.819285 0.096424  0.314108              -0.167191           1.307457
```

## Required corrective action

Do not present 0.928/0.967/0.096 as the primary prediagnostic model result. Freeze a prediction-time data dictionary, obtain source evidence for each B feature, exclude unresolved downstream/timing variables, prespecify one feature group and model-selection protocol, and evaluate it in a genuinely independent cohort. Until then, retain the result as an exploratory hypothesis-generating analysis alongside the frozen five-predictor V3 reference.

## Reproducibility evidence

Source hashes used by this audit:

```json
{
  "data\\interim\\validated\\z_alizadeh_sani.csv": "88a89ea916f6e2f62617d033a46fbd782e2e710f26569e53185679edcdcaca2b",
  "runner\\phase5\\zas_track\\run.py": "827109599e8a61466d2b7eaa567d91672feb823495c09775b2a3e0ecdbdf0b01",
  "runner\\phase5\\common\\models.py": "ccc24cb9aefbc621e41b8cd652fe6dd479708438bb223e38d17df547a924f56c",
  "runner\\phase5\\common\\metrics.py": "1b5b65f9d0a83cee3f29151ecf7d541ed13b9bb4473865b2ea6be85ceec21542"
}
```
