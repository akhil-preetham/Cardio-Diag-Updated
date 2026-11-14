# Feature Importance Audit

## Classification: Correct with limitations

Permutation importance is computed by `permutation_importance(best, x_test, y_test, ...)` inside each outer fold, so held-out labels are used for scoring but not for fitting, feature selection, or hyperparameter selection. The results are predictive associations, not causal effects.

The V2 artifact contains fold-level importance but does not itself provide a prespecified rank-stability analysis. `FEATURE_IMPORTANCE_STABILITY.csv` supplies descriptive means, standard deviations, positive fractions, and fold counts. Given the small test folds and correlated/repeated predictors, no substantive clinical feature conclusion is justified.
