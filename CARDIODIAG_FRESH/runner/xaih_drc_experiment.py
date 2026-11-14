from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cardiodiag.feature_selection import XAIHDRCSelector


def main(root: Path):
    output = root / "results/xaih_drc_experiments"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, filename, target in [
        ("five_states", "heart5.csv", "HeartDisease"),
        ("mendeley", "MendeleyCardiovascular_Disease_Dataset.csv", "target"),
    ]:
        frame = pd.read_csv(root / "reference/author_release" / filename)
        y = frame.pop(target).astype(int)
        identifiers = [c for c in frame if c.lower() in {"id", "patientid", "patient_id", "subject_id"} or c.lower().endswith("_id")]
        X = frame.drop(columns=identifiers)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        model = joblib.load(root / f"results/ensemble_experiments/{dataset}/corrected/osec/model.joblib")
        preprocessor = model.named_steps["preprocess"]
        fitted_model = model.named_steps["model"]
        train_matrix = preprocessor.transform(X_train)
        test_matrix = preprocessor.transform(X_test)
        feature_names = list(preprocessor.get_feature_names_out())
        explainer = LimeTabularExplainer(np.asarray(train_matrix), feature_names=feature_names, class_names=["negative", "positive"], mode="classification", random_state=42)
        values = {feature: [] for feature in feature_names}
        for index in range(min(50, len(train_matrix))):
            explanation = explainer.explain_instance(np.asarray(train_matrix[index]), fitted_model.predict_proba, num_samples=1000)
            for feature_index, weight in explanation.as_map()[1]:
                values[feature_names[feature_index]].append(abs(float(weight)))
        importance = {feature: float(np.mean(weights)) if weights else 0.0 for feature, weights in values.items()}
        selector = XAIHDRCSelector(n_features=min(11, len(feature_names)), pca_components=9).fit(
            pd.DataFrame(train_matrix, columns=feature_names),
            y_train.to_numpy(),
            lime_importances=importance,
        )
        selected_train = selector.transform(pd.DataFrame(train_matrix, columns=feature_names))
        selected_test = selector.transform(pd.DataFrame(test_matrix, columns=feature_names))
        downstream = LogisticRegression(max_iter=2000, random_state=42).fit(selected_train, y_train)
        downstream_probability = downstream.predict_proba(selected_test)[:, 1]
        downstream_prediction = (downstream_probability >= 0.5).astype(int)
        from cardiodiag.evaluation import classification_metrics

        downstream_metrics = classification_metrics(y_test, downstream_prediction, downstream_probability)
        (output / f"{dataset}_feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
        records.append({"dataset": dataset, "train_rows": len(train_matrix), "test_rows": len(test_matrix), "selected_features": selector.selected_features_, "pca_features": selector.pca_features_, "output_features": list(selected_train.columns), "train_shape": list(selected_train.shape), "test_shape": list(selected_test.shape), "lime_observations": min(50, len(train_matrix)), "downstream_model": "LogisticRegression", "downstream_metrics": downstream_metrics})
    (output / "xaih_drc_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
