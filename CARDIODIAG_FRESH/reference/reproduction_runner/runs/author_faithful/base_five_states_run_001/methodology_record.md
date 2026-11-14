# Author-faithful methodology record

This record preserves the source methodology and does not correct it.

- Categorical features are label-encoded before the train/test split.
- SMOTE is applied to the complete dataset before splitting.
- The split is 80:20 with `random_state=42`.
- Grid searches use `GridSearchCV` with `cv=5` and `scoring='accuracy'`.
- The source defines Decision Tree, Random Forest, Gradient Boosting, KNN, Logistic Regression, SVM, Naive Bayes, AdaBoost, and XGBoost-related models.
- A stacking classifier uses base estimators and Logistic Regression as the final estimator.
- Stacking is calibrated with `CalibratedClassifierCV(..., method='sigmoid', cv=5)`.
- The source repeatedly evaluates the held-out test set and uses test accuracy in its early-stopping loop.
- Reported metrics are not accepted as reproduced metrics until independently recovered from an approved run.

## Defects versus blockers

Methodological defects include preprocessing/SMOTE leakage, non-nested selection, and test-set-driven stopping. Execution blockers include unavailable Colab imports/paths, legacy or invalid grid candidates, and the XGBRegressor/classification-scoring mismatch. None is silently corrected in this track.
