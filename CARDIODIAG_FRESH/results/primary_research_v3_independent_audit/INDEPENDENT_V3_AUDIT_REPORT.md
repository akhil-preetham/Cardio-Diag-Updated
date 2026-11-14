# Independent V3 Audit Report

## Overall classification: Verified correct with limitations

The exact Stage 6 outer and inner assignments are reproduced. For every model, repeat, outer fold, and calibration fold, the recorded calibration fold is disjoint from the selection data and from every selection CV train/validation partition. Thus calibration-fold labels cannot influence the hyperparameters used to generate that fold's calibration score.

All independently recomputed probability and discrimination metrics match the saved V3 results. SVM native decision scores are not assigned Brier or log-loss metrics. The design remains limited by small calibration samples, repeated participants, unavailable patient identifiers, and conditional uncertainty.

The final estimator is single and refitted on full outer training data, while the fold-specific selected models are used only for calibration-score generation. Score-scale differences between those models and the final estimator remain a calibration limitation, not an outer-test leakage mechanism.
