from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cardiodiag.imbalance import loras_resample


def main(root: Path):
    output = root / "results/data_integrity_and_loras"
    output.mkdir(parents=True, exist_ok=True)
    report = []
    datasets = [
        ("five_states", "heart5.csv", "HeartDisease"),
        ("mendeley", "MendeleyCardiovascular_Disease_Dataset.csv", "target"),
    ]
    for name, filename, target in datasets:
        path = root / "reference/author_release" / filename
        frame = pd.read_csv(path)
        y = frame[target].astype(int)
        features = frame.drop(columns=[target])
        identifiers = [
            column
            for column in features
            if column.lower() in {"id", "patientid", "patient_id", "subject_id"}
            or column.lower().endswith("_id")
        ]
        features = features.drop(columns=identifiers)
        numeric = features.select_dtypes(include="number")
        X_train, _, y_train, y_test = train_test_split(
            features, y, test_size=0.25, stratify=y, random_state=42
        )
        train_numeric = pd.get_dummies(X_train, dtype=float).fillna(0.0)
        train_numeric = StandardScaler().fit_transform(train_numeric)
        loras_x, loras_y = loras_resample(
            train_numeric,
            y_train.to_numpy(),
            random_state=42,
            target_count=int(y_train.value_counts().max()),
        )
        smote_x, smote_y = SMOTE(random_state=42).fit_resample(train_numeric, y_train)
        report.append(
            {
                "dataset": name,
                "rows": len(frame),
                "features": features.shape[1],
                "identifier_columns": identifiers,
                "duplicate_rows": int(frame.duplicated().sum()),
                "duplicate_feature_rows": int(features.duplicated().sum()),
                "rounded_numeric_duplicates_3dp": int(numeric.round(3).duplicated().sum()),
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "loras_train_rows": len(loras_y),
                "loras_class_counts": pd.Series(loras_y).value_counts().to_dict(),
                "smote_train_rows": len(smote_y),
                "smote_class_counts": pd.Series(smote_y).value_counts().to_dict(),
                "loras_vs_smote_feature_shape": [list(loras_x.shape), list(smote_x.shape)],
            }
        )
    (output / "dataset_integrity_loras_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
