# Phase A — Model identity audit

The Phase 5 ZAS benchmark has 6 feature groups × 6 labels = 36 rows. `logistic` and `ridge` are exact aliases in both source and outputs: the constructor sets both to L2 `LogisticRegression`, `solver="lbfgs"`, `max_iter=3000`, the same seed and the same `model__C` values `[0.1, 1, 10]`. Their saved probabilities are identical at every repeated outer-test key and their metric rows match exactly.

The remaining Phase 5 label families are elastic-net logistic regression, spline logistic regression, random forest and histogram gradient boosting. The benchmark has at most five distinct estimator families, not six. This audit does not merge or rewrite the historical benchmark table.

The 0.928 result was not selected by a programmatic winner-selection rule. The ZAS runner creates all feature-group/model candidate predictions and benchmark rows. A human or report reader can select a row after seeing the results, but such selection does not become an unbiased evaluation simply because each candidate has nested hyperparameter selection. Feature group and estimator family were not selected inside an outer development loop. The fixed-prediction bootstrap flag in `ZAS_OPTIMISM_ANALYSIS.csv` explicitly says full model-development optimism was not corrected.

The V3 frozen model is a separate baseline protocol. Its fully nested calibration experiment and five-feature list should remain the primary historical reference. Phase 5's 4×2 repeated outer CV with native probabilities is not directly comparable to V3's calibrated 5-fold results as if it were a randomized head-to-head comparison.
