"""Generate LIME explanations from persisted ensemble models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split


def _raw(root: Path, dataset: str):
    path = root / ("reference/author_release/heart5.csv" if dataset == "five_states" else "reference/author_release/MendeleyCardiovascular_Disease_Dataset.csv")
    frame = pd.read_csv(path)
    target = "HeartDisease" if dataset == "five_states" else "target"
    y = frame.pop(target).astype(int)
    ids = [c for c in frame if c.lower() in {"id", "patientid", "patient_id", "subject_id"} or c.lower().endswith("_id")]
    return path, frame.drop(columns=ids), y


def _transformed(model, X_train, X_test, track):
    if track == "corrected":
        preprocessor = model.named_steps["preprocess"]
        return (
            preprocessor.transform(X_train),
            preprocessor.transform(X_test),
            list(preprocessor.get_feature_names_out()),
            model.named_steps["model"].predict_proba,
        )
    encoded_train = pd.get_dummies(X_train, dtype=float)
    encoded_test = pd.get_dummies(X_test, dtype=float).reindex(columns=encoded_train.columns, fill_value=0.0)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(encoded_train)
    return (
        scaler.transform(encoded_train),
        scaler.transform(encoded_test),
        list(encoded_train.columns),
        model.predict_proba,
    )


def main(root: Path, datasets: list[str], tracks: list[str], methods: list[str]):
    output = root / "results/lime_experiments"
    manifests = []
    for dataset in datasets:
        _, X, y = _raw(root, dataset)
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        for track in tracks:
            for method in methods:
                model_path = root / f"results/ensemble_experiments/{dataset}/{track}/{method}/model.joblib"
                if not model_path.exists():
                    continue
                model = joblib.load(model_path)
                train_matrix, test_matrix, feature_names, predict_proba = _transformed(model, X_train, X_test, track)
                probabilities = predict_proba(test_matrix)[:, 1]
                selected = [int(np.argmax(probabilities)), int(np.argmin(probabilities))]
                run_dir = output / dataset / track / method
                run_dir.mkdir(parents=True, exist_ok=True)
                explainer = LimeTabularExplainer(
                    np.asarray(train_matrix),
                    feature_names=feature_names,
                    class_names=["negative", "positive"],
                    mode="classification",
                    random_state=42,
                )
                rows = []
                for rank, index in enumerate(selected):
                    explanation = explainer.explain_instance(
                        np.asarray(test_matrix[index]),
                        predict_proba,
                        num_features=min(20, len(feature_names)),
                        num_samples=5000,
                    )
                    label = "positive_representative" if rank == 0 else "negative_representative"
                    explanation.save_to_file(str(run_dir / f"{label}.html"))
                    for feature, weight in explanation.as_list(label=1):
                        rows.append({"observation": label, "test_index": index, "feature": feature, "weight": weight, "probability": probabilities[index]})
                pd.DataFrame(rows).to_csv(run_dir / "explanations.csv", index=False)
                manifests.append({"dataset": dataset, "track": track, "method": method, "model": str(model_path), "n_features": len(feature_names), "representative_indices": selected})
    (output / "lime_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=["five_states", "mendeley"])
    parser.add_argument("--tracks", nargs="+", default=["author_faithful", "compatibility_adapted", "corrected"])
    parser.add_argument("--methods", nargs="+", default=["ovec", "osec"])
    args = parser.parse_args()
    main(args.root, args.datasets, args.tracks, args.methods)
