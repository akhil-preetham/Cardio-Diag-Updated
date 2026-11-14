# Compatibility remediation plan

## Scope and status

This is a planning-only document. No notebook, kernel, model, or `GridSearchCV` was executed. No package was installed or changed, and neither the original nor prepared notebook was modified.

The prepared working copy remains distinct from the original. Its only existing changes are the environment-only cell 2 and cell 4 changes documented in `base_notebook_patch_manifest.json`.

## Compatibility blockers

### Cell 13: Decision Tree grid

```python
'max_features': ['auto', 'sqrt', 'log2', None]
```

`auto` is a legacy value rejected by current scikit-learn DecisionTreeClassifier parameter validation. Removing it or translating it to `sqrt` changes the searched parameter space. This is implementation compatibility (B), not an exact reproduction.

### Cell 13: KNN grid

```python
'metric': ['minkowskii', 'euclidean', 'manhattan']
```

`minkowskii` is misspelled. Correcting it to `minkowski` is probably the intended algorithm, but it changes the candidate grid and must remain in a separate compatibility copy.

### Cell 13: Logistic Regression grid

```python
'penalty': ['l1', 'l2', 'none'],
'solver': ['liblinear', 'lbfgs']
```

Several combinations are invalid: `liblinear` does not support all listed penalties, `lbfgs` does not support `l1`, and the legacy string `none` is not the current API representation. GridSearchCV may report failed candidates or abort depending on its error handling. Any filtering changes the search space.

### Cell 13: XGBoost legacy parameters

```python
'objective': ['reg:linear'],
'silent': [1]
```

These are legacy XGBoost parameters. A current-compatible objective such as `reg:squarederror` and a supported verbosity option would be an implementation change, not a faithful reproduction.

### Cell 15: Regressor scored as a classifier

```python
xgb_reg = XGBRegressor()
grid_search_xgb_reg = GridSearchCV(xgb_reg, param_grid_xgb_reg, cv=5, scoring='accuracy')
```

This is a methodological incompatibility, not merely an API issue. A regressor produces continuous outputs while `accuracy` is a classification scorer. Replacing the estimator with `XGBClassifier` or changing the metric changes the experiment and belongs only in corrected validation or a separately labelled compatibility experiment.

### Cells 9–11: methodological defects, not compatibility fixes

```python
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)
```

SMOTE occurs before splitting and the split is 80:20. These must remain unchanged in an author-faithful track. Leakage-safe resampling and a 75:25 comparator belong exclusively in corrected validation.

### Cell 19: test-set-driven stopping

```python
calibrated_stacking.fit(X_train, y_train)
y_pred = calibrated_stacking.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

Test accuracy controls the stopping loop. This is a methodological defect and must not be silently corrected in the faithful copy.

## Remediation classifications

### A — Environment compatibility

- Cell 2: omit the unavailable Colab Drive mount in the prepared copy.
- Cell 4: replace the Colab Drive path with the verified local `heart5.csv`.

These changes preserve dataset identity and experimental operations. They remain author-faithful environment adaptations.

### B — Implementation compatibility

- Cell 13 Decision Tree `auto`.
- Cell 13 KNN `minkowskii`.
- Cell 13 Logistic Regression penalty/solver grid.
- Cell 13 legacy XGBoost objective and verbosity parameters.

These changes may preserve the intended estimator family, but they alter the implementation or search space. They must be applied only to a new compatibility-adapted copy. Results cannot be called exact author-faithful reproduction.

### C — Methodological correction

- Cell 9–11 SMOTE placement and split protocol.
- Cell 5 preprocessing provenance.
- Cell 15 regressor/classification scoring mismatch.
- Cell 19 test-set-driven stopping.

These change data processing, validation, model selection, estimator/metric meaning, or scientific conclusions. They belong exclusively in corrected validation.

### D — Unresolved issues

- The exact historical environment is not supplied.
- No version combination has been proven compatible without executing the notebook.
- Current versions are only partially compatible with the source grid.
- The faithful result may be a controlled execution failure at the first incompatible cell.

## Version strategy

The current `.venv-repro` should not be modified. The preferred strategy is a separate, disposable historical-environment candidate, but it must be treated as unverified until package metadata and API checks are completed.

| Candidate | Suggested pins/ranges | Confidence |
|---|---|---|
| Current environment | Python 3.11.16; NumPy 2.4.6; pandas 3.1.1; scikit-learn 1.9.1; imbalanced-learn 0.14.2; XGBoost 3.2.0; matplotlib 3.11.2 | Installed, but only partially compatible |
| Historical-style candidate | Python 3.8 or 3.9; NumPy 1.23.x; pandas 1.5.x; scikit-learn 1.1.x; imbalanced-learn 0.9.x; XGBoost 1.6.x; matplotlib 3.6.x | Unverified hypothesis; no installation performed |
| Minimal compatibility candidate | Keep current environment and adapt only invalid API/grid values in a new copy | Technically plausible, but not exact reproduction |

The historical-style candidate may accept more legacy syntax, but `minkowskii` is a spelling error rather than a version issue, and `XGBRegressor` with classification accuracy remains conceptually invalid. Older packages may also require Python 3.8/3.9 and cannot be assumed compatible with Python 3.11.

## Faithful versus corrected tracks

### Author-faithful

Preserve cells 5 and 9–20 exactly, including SMOTE-before-split, the 80:20 split, invalid grid candidates, XGBRegressor classification scoring, calibration, and test-set-driven stopping. If the notebook fails, record the exact cell and traceback. Do not repair it silently.

### Compatibility-adapted

Use a new copied notebook and apply only explicitly recorded B changes. Label results as compatibility-adapted, not author-faithful. Do not merge its metrics with faithful or corrected results.

### Corrected validation

Use an independent implementation with training-fold preprocessing, split-before-SMOTE, leakage-safe CV, no test-set selection, appropriate estimator/metric pairing, and documented seeds. These results answer a different scientific question.

## Safety verification

- Original notebook unchanged.
- Prepared working notebook unchanged during this planning phase.
- Original author repository clean.
- Notebook execution remains disabled.
- No packages changed.
- No model, kernel, or notebook execution occurred.

## Final recommendation

Do not execute the prepared notebook yet. First decide whether the next controlled attempt should intentionally record the faithful failure, or whether a separately copied compatibility-adapted notebook should be prepared. Any compatibility-adapted result must remain clearly separate from author-faithful reproduction and corrected validation.
