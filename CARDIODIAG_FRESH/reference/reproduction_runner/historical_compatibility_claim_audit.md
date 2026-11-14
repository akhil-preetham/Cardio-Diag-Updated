# Historical compatibility claim audit

This is a static audit. It does not establish a historical environment and does not execute the notebook.

| Expression | Cell | Claim classification | Conclusion |
|---|---:|---|---|
| `max_features: ['auto', 'sqrt', 'log2', None]` | 13 | Current-version incompatibility; historical behavior unverified | Current scikit-learn parameter validation rejects the legacy `auto` value. No supplied artifact proves which older release accepted it. |
| `metric: ['minkowskii', 'euclidean', 'manhattan']` | 13 | Likely source-level error | `minkowskii` is not the standard spelling `minkowski`. No reasonable version claim can be made that this typo is supported. |
| `penalty: ['l1','l2','none']`, `solver: ['liblinear','lbfgs']` | 13 | Verified API constraints plus unverified historical behavior | Solver capabilities make several combinations invalid. Historical treatment of the `none` spelling is not established by supplied metadata. |
| `objective: ['reg:linear']` | 13 | Current-version legacy incompatibility; historical support unverified | The installed XGBoost line uses modern objective naming/defaults. No exact historical release supporting the source is supplied. |
| `silent: [1]` | 13 | Current-version legacy compatibility unverified | It is a legacy verbosity argument. Static records do not prove whether the installed version rejects, ignores, or aliases it in this grid. |
| `XGBRegressor` with `scoring='accuracy'` | 15 | Source-level methodological incompatibility | A continuous regressor is paired with a classification scorer. Version selection cannot make this scientifically equivalent to classification. |
| `CalibratedClassifierCV(stacking_classifier, method='sigmoid', cv=5)` | 18 | Current-version compatible; historical behavior unverified | The current API accepts the positional estimator form. This does not certify every historical release. |
| `StackingClassifier(..., final_estimator=...)` | 17 | Current-version compatible; historical behavior unverified | The current API supports the source structure. Historical release support is not established. |

## Overall answer

No single Python environment is proven by the supplied evidence to execute the original source completely without changing model definitions or parameter grids. A historical environment might preserve some legacy API values, but that is an unverified hypothesis and would not repair the KNN spelling or regressor/classification scoring mismatch.

## Preservation consequence

The author-faithful track must preserve these expressions and may legitimately end in a controlled compatibility failure. Any B-category adaptation belongs in a separate copy; the regressor/scoring correction belongs to corrected validation.
