from __future__ import annotations

from collections.abc import Mapping

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from cardiodiag.ensembles import build_osec, build_ovec


def optimize_ovec(
    X,
    y,
    *,
    parameter_distributions: Mapping[str, object],
    n_iter: int = 10,
    cv: int = 5,
    random_state: int = 42,
    scoring: str = "accuracy",
):
    """Run the source-compatible OVEC random search on training data only."""
    search = RandomizedSearchCV(
        build_ovec(random_state=random_state),
        parameter_distributions=parameter_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    return search.fit(X, y)


def optimize_osec(
    X,
    y,
    *,
    parameter_grid: Mapping[str, object],
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
):
    """Run the source-compatible OSEC grid search on training data only."""
    search = GridSearchCV(
        build_osec(random_state=random_state, cv=cv),
        param_grid=parameter_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    return search.fit(X, y)
