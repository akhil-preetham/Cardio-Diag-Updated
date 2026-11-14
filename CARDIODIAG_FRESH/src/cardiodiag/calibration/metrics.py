from __future__ import annotations

from itertools import pairwise

import numpy as np


def calibration_summary(y_true, probabilities, *, bins: int = 10) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have identical lengths")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    nonempty = 0
    for index, (lower, upper) in enumerate(pairwise(edges)):
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not mask.any():
            continue
        gap = abs(float(probabilities[mask].mean()) - float(y_true[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
        nonempty += 1
    return {"ece": ece, "mce": mce, "bins": bins, "nonempty_bins": nonempty}
