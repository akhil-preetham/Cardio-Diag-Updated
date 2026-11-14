from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from cardiodiag.fairness import subgroup_metrics


def main(root: Path):
    output = root / "results/ensemble_postprocessing"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, filename, target, subgroup in [
        ("five_states", "heart5.csv", "HeartDisease", "Sex"),
        ("mendeley", "MendeleyCardiovascular_Disease_Dataset.csv", "target", "gender"),
    ]:
        frame = pd.read_csv(root / "reference/author_release" / filename)
        y = frame.pop(target).astype(int)
        group_values = frame.pop(subgroup)
        identifier = "patientid" if "patientid" in frame.columns else None
        if identifier:
            frame = frame.drop(columns=[identifier])
        _, X_test, _, group_test = train_test_split(
            frame, group_values, test_size=0.25, stratify=y, random_state=42
        )
        del X_test
        for track in ["corrected"]:
            for method in ["ovec", "osec"]:
                prediction_path = root / f"results/ensemble_experiments/{dataset}/{track}/{method}/predictions.csv"
                if not prediction_path.exists():
                    continue
                prediction = pd.read_csv(prediction_path)
                metrics = subgroup_metrics(
                    prediction["y_true"],
                    prediction["prediction"],
                    prediction["probability"],
                    group_test.reset_index(drop=True),
                    minimum_size=20,
                )
                if not metrics.empty:
                    metrics.insert(0, "dataset", dataset)
                    metrics.insert(1, "track", track)
                    metrics.insert(2, "method", method)
                    rows.append(metrics)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.to_csv(output / "ensemble_subgroup_performance.csv", index=False)
    (output / "fairness_manifest.txt").write_text(
        "Descriptive subgroup performance only; minimum subgroup size=20. "
        "No fairness certification or causal claim is made.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
