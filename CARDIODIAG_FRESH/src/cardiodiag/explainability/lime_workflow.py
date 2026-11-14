from __future__ import annotations

from pathlib import Path


class LimeUnavailableError(RuntimeError):
    """Raised when the optional LIME dependency is unavailable."""


def explain_with_lime(
    predict_proba,
    training_data,
    observation,
    *,
    feature_names: list[str],
    class_names: list[str] | None = None,
    random_state: int = 42,
    num_samples: int = 5000,
    output_html: str | Path | None = None,
):
    """Generate one tabular LIME explanation from a supplied fitted pipeline.

    The dependency is optional. The function refuses to silently substitute a
    different explanation method when LIME is unavailable.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise LimeUnavailableError(
            "LIME is not installed in the selected environment; no explanation was generated."
        ) from exc
    explainer = LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=class_names or ["negative", "positive"],
        mode="classification",
        random_state=random_state,
    )
    explanation = explainer.explain_instance(
        observation,
        predict_proba,
        num_samples=num_samples,
    )
    if output_html is not None:
        explanation.save_to_file(str(output_html))
    return explanation
