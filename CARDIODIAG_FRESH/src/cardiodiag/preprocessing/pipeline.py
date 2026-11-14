from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def identify_columns(
    columns: Sequence[str],
    *,
    categorical: Sequence[str] = (
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ),
) -> tuple[list[str], list[str]]:
    categorical_set = set(columns) & set(categorical)
    categorical_columns = [column for column in columns if column in categorical_set]
    numeric_columns = [column for column in columns if column not in categorical_set]
    return numeric_columns, categorical_columns


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_columns)),
            ("categorical", categorical, list(categorical_columns)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
