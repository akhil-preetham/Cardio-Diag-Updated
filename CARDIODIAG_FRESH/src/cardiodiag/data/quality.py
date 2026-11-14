from __future__ import annotations

from typing import Any

import pandas as pd


def identifier_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.lower() in {"id", "patientid", "patient_id", "subject_id"}
        or column.lower().endswith("_id")
    ]


def duplicate_summary(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    predictors = [
        column
        for column in frame.columns
        if column != target and column not in identifier_columns(frame)
    ]
    exact_groups = frame.groupby(list(frame.columns), dropna=False).size()
    predictor_groups = frame.groupby(predictors, dropna=False).size()
    conflicting = 0
    for _, group in frame.groupby(predictors, dropna=False):
        if len(group) > 1 and group[target].nunique(dropna=False) > 1:
            conflicting += 1
    return {
        "exact_duplicate_occurrences": int(frame.duplicated(keep="first").sum()),
        "exact_duplicate_groups": int((exact_groups > 1).sum()),
        "predictor_duplicate_occurrences": int(frame.duplicated(predictors, keep="first").sum()),
        "predictor_duplicate_groups": int((predictor_groups > 1).sum()),
        "conflicting_predictor_groups": conflicting,
    }


def quality_summary(
    frame: pd.DataFrame,
    *,
    target: str,
    dataset: str,
    implausible: dict[str, int] | None = None,
) -> dict[str, Any]:
    duplicate = duplicate_summary(frame, target)
    return {
        "dataset": dataset,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "target": target,
        "target_values": {
            str(key): int(value)
            for key, value in frame[target].value_counts(dropna=False).sort_index().items()
        },
        "target_percentages": {
            str(key): round(float(value / len(frame) * 100), 6)
            for key, value in frame[target].value_counts(dropna=False).sort_index().items()
        },
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "unique_values": {
            column: [str(value) for value in frame[column].drop_duplicates().head(100).tolist()]
            for column in frame.select_dtypes(exclude="number").columns
        },
        "numeric_ranges": {
            column: {
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
                "q01": float(frame[column].quantile(0.01)),
                "q50": float(frame[column].quantile(0.50)),
                "q99": float(frame[column].quantile(0.99)),
            }
            for column in frame.select_dtypes(include="number").columns
        },
        "identifier_columns": identifier_columns(frame),
        "constant_columns": [
            column for column in frame.columns if frame[column].nunique(dropna=False) <= 1
        ],
        "implausible_value_counts": implausible or {},
        "duplicates": duplicate,
    }
