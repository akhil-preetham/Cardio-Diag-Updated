# CARDIODIAG methodology record

The released base notebook label-encodes categorical features, applies SMOTE before the 80:20 split, performs five-fold GridSearchCV with accuracy scoring, builds a stacking classifier, calibrates it with sigmoid calibration, and repeatedly evaluates held-out test accuracy for stopping. These operations were preserved in the author-faithful run.

The corrected comparator uses a stratified 75:25 split, training-fold preprocessing, SMOTE inside an imbalanced-learn pipeline, five-fold stratified model selection on ROC-AUC, and post-selection test evaluation. Mendeley `patientid` is excluded as an identifier.
