# Historical environment feasibility

## Scope

This is a static feasibility assessment. No notebook, kernel, model, `GridSearchCV`, package installation, or environment modification was performed.

The inspected source is the original `Daza_Paper_Exact_Accuracy.ipynb`; the prepared copy was not modified during this assessment.

## Evidence available locally

The selected environment currently contains:

- Python 3.11.16
- NumPy 2.4.6
- pandas 3.1.1
- scikit-learn 1.9.1
- imbalanced-learn 0.14.2
- XGBoost 3.2.0
- matplotlib 3.11.2

Static API inspection confirmed:

- Current DecisionTreeClassifier parameter constraints do not include the legacy string `auto` for `max_features`.
- Current `CalibratedClassifierCV` accepts the notebook's positional estimator form.
- Current `StackingClassifier` accepts the notebook's estimator list/final-estimator structure.
- Current XGBRegressor defaults to `reg:squarederror`; the notebook requests legacy `reg:linear`.
- Current XGBRegressor exposes a legacy `silent` parameter only as a compatibility placeholder/default, not as evidence that the old grid is semantically preserved.
- The notebook's `accuracy` scorer is a classification scorer, while XGBRegressor is a regressor.

The exact historical author environment is not included in the repository. Therefore, exact historical version claims remain unverified.

## Feasibility by component

| Component | Historical source | Feasibility without source changes | Evidence conclusion |
|---|---|---:|---|
| Decision Tree grid | `max_features=['auto', 'sqrt', 'log2', None]` | Unverified | `auto` is rejected by current scikit-learn. An older scikit-learn may have accepted it, but no local historical metadata proves which release. |
| KNN grid | `metric=['minkowskii', 'euclidean', 'manhattan']` | Not established; likely impossible as written | `minkowskii` is a spelling error. Version selection cannot be assumed to make that metric valid. |
| Logistic Regression grid | `penalty=['l1','l2','none']`, `solver=['liblinear','lbfgs']` | Partially feasible | Some combinations are invalid by solver capability in any ordinary scikit-learn API; historical handling of the string `none` is version-dependent and unverified. |
| XGBoost grid | `objective='reg:linear'`, `silent=1` | Historically plausible, locally unverified | Older XGBoost may have accepted legacy parameters. No local package metadata proves the exact compatible release. |
| XGBRegressor + accuracy | `GridSearchCV(..., scoring='accuracy')` | Not scientifically valid as written | This is an estimator/metric mismatch, not a version-only problem. A source or metric change is unavoidable for a meaningful successful experiment. |
| Calibration | `CalibratedClassifierCV(stacking_classifier, method='sigmoid', cv=5)` | Feasible in current API | Current signature accepts a positional estimator; no compatibility change is indicated. |
| Stacking | `StackingClassifier(..., final_estimator=best_logistic, cv=5)` | Feasible in current API | Current signature supports the source structure. Runtime behavior and estimator validity remain untested. |

## Category B resolution

1. **Decision Tree `auto`:** an older dependency may preserve the source, but the exact release is unverified. Minimal grid adaptation is possible but changes the implementation.
2. **KNN `minkowskii`:** not a credible version-only blocker; spelling correction is unavoidable if that candidate is intended to run.
3. **Logistic Regression grid:** no dependency version can make every solver/penalty combination mathematically valid. Invalid combinations must either be allowed to fail or removed in a compatibility-adapted copy.
4. **XGBoost legacy parameters:** an older version may accept them, but the required historical version is not available locally. Translating them is an implementation compatibility change.
5. **CalibratedClassifierCV and StackingClassifier:** no Category B remediation is currently required based on static API evidence.

## Version strategy

The current `.venv-repro` must remain unchanged.

A separate historical-style environment could be investigated with a candidate such as Python 3.8/3.9, NumPy 1.23.x, pandas 1.5.x, scikit-learn 1.1.x, imbalanced-learn 0.9.x, XGBoost 1.6.x, and matplotlib 3.6.x. This is a **version hypothesis, not a verified compatibility claim**. It may preserve more legacy syntax, but it cannot fix the KNN spelling error or the regressor/classification metric mismatch.

The preferred evidence order is:

1. Attempt a faithful run only if the exact historical environment can be independently established.
2. Otherwise record the faithful failure without changing the source.
3. Prepare a separate compatibility-adapted copy for B changes.
4. Keep all C corrections in corrected validation only.

## Conclusion

The notebook cannot currently be certified as executable without scientific or implementation changes. Calibration and stacking are statically compatible; the Decision Tree and XGBoost legacy APIs are historically plausible but unverified; KNN spelling and XGBRegressor classification scoring remain blockers independent of ordinary version selection.
