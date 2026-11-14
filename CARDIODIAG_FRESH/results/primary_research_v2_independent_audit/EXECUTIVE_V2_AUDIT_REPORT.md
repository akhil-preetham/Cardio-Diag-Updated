# Executive V2 Independent Audit Report

## Overall classification: Requires correction

The persisted outer and inner split assignments are exactly reproduced, the outer-test predictions are complete, and all independently recomputed metrics and calibration diagnostics match the saved V2 tables. SVM scores are correctly labelled as decision scores, and invalid native SVM Brier/log-loss summaries are not reported.

The principal unresolved methodological issue is calibration nesting. V2 generates inner cross-fitted calibration scores, but selects the hyperparameters used for those scores using all four inner folds, including the calibration fold. This creates selection-induced optimism in the calibration-training scores. It is not outer-test leakage, but it requires a fully nested calibration correction before V2 can be accepted as the final calibrated baseline.

Uncertainty and statistical comparisons are appropriately labelled descriptive conditional resampling analyses. Feature importance is held-out predictive permutation importance, but it does not support causal or substantive clinical claims. The separate Version 2 results update is numerically accurate and avoids external-validation or clinical-utility overclaims.

No V2 model training or cross-validation was rerun during this audit, and historical artifacts were preserved.
