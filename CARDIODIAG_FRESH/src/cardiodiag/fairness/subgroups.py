from __future__ import annotations

import pandas as pd

from cardiodiag.evaluation import classification_metrics


def subgroup_metrics(
    labels: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
    subgroup: pd.Series,
    *,
    minimum_size: int = 20,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"y_true": labels, "y_pred": predictions, "probability": probabilities, "group": subgroup}
    ).dropna(subset=["group"])
    rows = []
    for group, values in frame.groupby("group", dropna=False):
        if len(values) < minimum_size:
            continue
        metrics = classification_metrics(values.y_true, values.y_pred, values.probability)
        rows.append({"group": group, "n": len(values), **metrics})
    return pd.DataFrame(rows)
