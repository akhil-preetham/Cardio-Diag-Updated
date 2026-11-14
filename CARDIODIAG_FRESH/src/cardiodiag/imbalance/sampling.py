from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


class LoRASUnderSpecifiedError(RuntimeError):
    """Raised when the optional, source-used LoRAS package is unavailable."""


def build_smote_pipeline(steps: Sequence[tuple[str, object]], *, random_state: int = 42) -> Pipeline:
    """Insert SMOTE into a training-only imbalanced-learn pipeline.

    The sampler is fitted by the pipeline on each training fold; callers must
    never call it on a complete dataset before partitioning.
    """
    return Pipeline([*steps, ("smote", SMOTE(random_state=random_state))])


def loras_resample(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
    target_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact package API used by the released notebooks.

    The author source calls ``loras.fit_resample(majority, minority)`` and
    then optionally downsamples the returned minority class to the majority
    count. This wrapper preserves that behavior and returns a complete
    feature/label pair. It must be called only on a training partition.
    """
    try:
        import loras
    except ImportError as exc:
        raise LoRASUnderSpecifiedError(
            "The source-used loras==0.0.22b0 package is unavailable."
        ) from exc
    features = np.asarray(features)
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("LoRAS resampling requires exactly two classes")
    minority = classes[np.argmin(counts)]
    majority = classes[np.argmax(counts)]
    majority_features = features[labels == majority]
    minority_features = features[labels == minority]
    desired = target_count if target_count is not None else len(majority_features)
    generated = loras.fit_resample(
        majority_features,
        minority_features,
        num_generated_points=max(0, desired - len(minority_features)),
        random_state=random_state,
    )
    if len(generated) > desired:
        rng = np.random.default_rng(random_state)
        generated = generated[rng.choice(len(generated), size=desired, replace=False)]
    output_features = np.concatenate((majority_features, generated), axis=0)
    output_labels = np.concatenate(
        (
            np.full(len(majority_features), majority),
            np.full(len(generated), minority),
        )
    )
    return output_features, output_labels
