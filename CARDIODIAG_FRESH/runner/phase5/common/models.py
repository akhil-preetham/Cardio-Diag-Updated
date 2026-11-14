"""Fold-local model constructors used by Phase 5."""
from __future__ import annotations

from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def estimator(model: str, features: list[str], dataframe) -> tuple[Pipeline, dict[str, list[object]]]:
    numeric = [column for column in features if is_numeric_dtype(dataframe[column])]
    categorical = [column for column in features if column not in numeric]
    if model == "spline_logistic":
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("spline", SplineTransformer(n_knots=3, degree=2)),
            ("scale", StandardScaler()),
        ]
    else:
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )
    if model in {"logistic", "ridge", "elastic_net", "spline_logistic"}:
        penalty = "l1" if model == "elastic_net" else "l2"
        solver = "saga" if penalty == "l1" else "lbfgs"
        classifier = LogisticRegression(penalty=penalty, solver=solver, max_iter=3000, random_state=5107)
        grid = {"model__C": [0.1, 1.0, 10.0]}
        if model == "elastic_net":
            classifier = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=3000, random_state=5107)
            grid = {"model__C": [0.1, 1.0, 10.0], "model__l1_ratio": [0.25, 0.75]}
    elif model == "random_forest":
        classifier = RandomForestClassifier(n_estimators=250, random_state=5107, n_jobs=1)
        grid = {"model__max_depth": [None, 5], "model__min_samples_leaf": [2, 8]}
    elif model == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(max_iter=120, random_state=5107)
        grid = {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [7, 15]}
    else:
        raise ValueError(model)
    return Pipeline([("preprocess", transformer), ("model", classifier)]), grid
