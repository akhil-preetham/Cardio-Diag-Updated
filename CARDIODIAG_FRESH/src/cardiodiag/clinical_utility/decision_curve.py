from __future__ import annotations

import numpy as np


def net_benefit(y_true, probabilities, threshold: float) -> float:
    """Calculate decision-curve net benefit for a prespecified threshold."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have identical lengths")
    predicted_positive = probabilities >= threshold
    tp = np.sum(predicted_positive & (y_true == 1))
    fp = np.sum(predicted_positive & (y_true == 0))
    n = len(y_true)
    odds = threshold / (1.0 - threshold)
    return float((tp / n) - (fp / n) * odds)
