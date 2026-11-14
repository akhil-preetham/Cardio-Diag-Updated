from __future__ import annotations

import numpy as np
from scipy.stats import norm


def paired_bootstrap_difference(
    y_true: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    *,
    metric,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap a paired metric difference on identical observations."""
    y_true = np.asarray(y_true)
    first_prediction = np.asarray(first_prediction)
    second_prediction = np.asarray(second_prediction)
    if not (len(y_true) == len(first_prediction) == len(second_prediction)):
        raise ValueError("Paired inputs must have identical lengths")
    rng = np.random.default_rng(random_state)
    observed = float(metric(y_true, first_prediction) - metric(y_true, second_prediction))
    estimates = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        estimates.append(metric(y_true[indices], first_prediction[indices]) - metric(y_true[indices], second_prediction[indices]))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {"difference": observed, "ci_low": float(low), "ci_high": float(high)}


def delong_auc_test(
    y_true: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
) -> dict[str, float]:
    """Compare two ROC-AUC values with the paired DeLong test.

    Both prediction vectors must describe the same observations in the same
    order. The implementation uses the midrank covariance formulation.
    """
    y_true = np.asarray(y_true)
    first_probability = np.asarray(first_probability, dtype=float)
    second_probability = np.asarray(second_probability, dtype=float)
    if not (len(y_true) == len(first_probability) == len(second_probability)):
        raise ValueError("DeLong inputs must have identical lengths")
    positive = y_true == 1
    negative = y_true == 0
    if not positive.any() or not negative.any():
        raise ValueError("DeLong requires both outcome classes")

    def midrank(values):
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
            start = end
        return ranks

    predictions = np.vstack([first_probability, second_probability])
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    aucs = np.empty(2)
    covariance = np.empty((2, 2))
    structural = []
    for prediction in predictions:
        positive_ranks = midrank(prediction[positive])
        negative_ranks = midrank(prediction[negative])
        combined = midrank(prediction)
        aucs[len(structural)] = (combined[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
        structural.append(
            (
                (combined[positive] - positive_ranks) / n_negative,
                (combined[negative] - negative_ranks) / n_positive,
            )
        )
    for left in range(2):
        for right in range(2):
            positive_cov = np.cov(structural[left][0], structural[right][0], ddof=1)[0, 1] / n_positive
            negative_cov = np.cov(structural[left][1], structural[right][1], ddof=1)[0, 1] / n_negative
            covariance[left, right] = positive_cov + negative_cov
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        return {"auc_first": float(aucs[0]), "auc_second": float(aucs[1]), "difference": difference, "z": 0.0, "p_value": 1.0}
    z_value = difference / np.sqrt(variance)
    return {
        "auc_first": float(aucs[0]),
        "auc_second": float(aucs[1]),
        "difference": difference,
        "z": float(z_value),
        "p_value": float(2 * norm.sf(abs(z_value))),
    }
