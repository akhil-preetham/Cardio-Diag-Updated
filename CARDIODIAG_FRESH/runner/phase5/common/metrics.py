"""Shared Phase 5 metrics and bootstrap helpers."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, _tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
    return {
        "n": len(y),
        "events": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, probability)),
        "sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def bootstrap_difference(y: np.ndarray, first: np.ndarray, second: np.ndarray, seed: int = 5107, repetitions: int = 400) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences = {name: [] for name in ["roc_auc", "pr_auc", "brier", "log_loss"]}
    for _ in range(repetitions):
        indexes = rng.integers(0, len(y), len(y))
        sample_y = y[indexes]
        if len(np.unique(sample_y)) < 2:
            continue
        first_probability = first[indexes]
        second_probability = second[indexes]
        left = {
            "roc_auc": roc_auc_score(sample_y, first_probability),
            "pr_auc": average_precision_score(sample_y, first_probability),
            "brier": brier_score_loss(sample_y, first_probability),
            "log_loss": log_loss(sample_y, np.clip(first_probability, 1e-6, 1 - 1e-6), labels=[0, 1]),
        }
        right = {
            "roc_auc": roc_auc_score(sample_y, second_probability),
            "pr_auc": average_precision_score(sample_y, second_probability),
            "brier": brier_score_loss(sample_y, second_probability),
            "log_loss": log_loss(sample_y, np.clip(second_probability, 1e-6, 1 - 1e-6), labels=[0, 1]),
        }
        for name, values in differences.items():
            values.append(right[name] - left[name])
    return {
        f"{name}_ci_low": float(np.quantile(values, 0.025))
        for name, values in differences.items()
    } | {
        f"{name}_ci_high": float(np.quantile(values, 0.975))
        for name, values in differences.items()
    }
