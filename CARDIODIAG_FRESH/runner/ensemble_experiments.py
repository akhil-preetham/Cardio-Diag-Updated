"""Controlled OVEC/OSEC experiments for the released datasets.

This runner is separate from notebook execution. It never discovers or
modifies notebooks and writes each dataset/track/method to a new run folder.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cardiodiag.data.loading import sha256_file
from cardiodiag.evaluation import classification_metrics

SEED = 42


def _dataset(path: Path, name: str) -> tuple[pd.DataFrame, pd.Series, str]:
    frame = pd.read_csv(path)
    target = "HeartDisease" if name == "five_states" else "target"
    y = pd.to_numeric(frame.pop(target), errors="raise").astype(int)
    identifiers = [c for c in frame if c.lower() in {"id", "patientid", "patient_id", "subject_id"} or c.lower().endswith("_id")]
    return frame.drop(columns=identifiers), y, target


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


class Pipeline:
    """Local alias kept to make the two transformer branches visually explicit."""

    def __new__(cls, steps):
        from sklearn.pipeline import Pipeline as SkPipeline

        return SkPipeline(steps)


def _author_transform(X: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(X, dtype=float)
    return pd.DataFrame(StandardScaler().fit_transform(encoded), columns=encoded.columns, index=X.index)


def _split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    track: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    if track in {"author_faithful", "compatibility_adapted"}:
        transformed = _author_transform(X)
        sampler = SMOTE(random_state=SEED)
        sampled_X, sampled_y = sampler.fit_resample(transformed, y)
        return train_test_split(sampled_X, sampled_y, test_size=0.2, random_state=SEED, stratify=sampled_y)
    return train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)


def _ovec_search(X_train, y_train, track: str):
    rf_estimators = [2, None] if track == "author_faithful" else [2]
    params = {
        "logistic": {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]},
        "rf": {"max_depth": [2, 5, None], "n_estimators": rf_estimators, "criterion": ["gini", "entropy"]},
        "knn": {"n_neighbors": [5, 7, 9, 11], "weights": ["uniform", "distance"], "metric": ["manhattan", "euclidean"]},
        "svm": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"], "kernel": ["rbf", "linear"]},
    }
    estimators = {
        "logistic": LogisticRegression(max_iter=2000, random_state=SEED),
        "rf": RandomForestClassifier(random_state=SEED, n_jobs=1),
        "knn": KNeighborsClassifier(),
        "svm": SVC(probability=True, random_state=SEED),
    }
    fitted = {}
    search_records = []
    for name in ["logistic", "rf", "knn", "svm"]:
        search = RandomizedSearchCV(
            estimators[name],
            params[name],
            n_iter=2,
            cv=5,
            scoring="accuracy",
            random_state=SEED,
            n_jobs=1,
            error_score="raise",
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        search_records.append({"model": name, "best_params": search.best_params_, "best_score": float(search.best_score_), "track": track})
    voting = "hard" if track == "author_faithful" else "soft"
    weights = [2, 2, 1, 2] if voting == "hard" else [1, 2, 1, 2]
    model = VotingClassifier(
        estimators=[(name, fitted[name]) for name in ["knn", "rf", "logistic", "svm"]],
        voting=voting,
        weights=weights,
        n_jobs=1,
    )
    return model, search_records


def _osec_search(X_train, y_train):
    grids = [
        ("tree", DecisionTreeClassifier(random_state=SEED), {
            "max_depth": [5], "max_features": ["sqrt", None], "min_samples_leaf": [2],
            "min_samples_split": [2], "criterion": ["entropy"], "splitter": ["best"],
        }),
        ("logistic", LogisticRegression(random_state=SEED, max_iter=2000), {
            "penalty": ["l2"], "C": np.logspace(-4, 4, 20), "solver": ["liblinear"],
        }),
        ("bayes", GaussianNB(), {"var_smoothing": np.logspace(0, -9, 100)}),
        ("gradient", GradientBoostingClassifier(random_state=SEED), {
            "n_estimators": [75, 100, 125], "learning_rate": [1.0, 0.1, 0.001, 0.0001],
        }),
    ]
    fitted = {}
    records = []
    for name, estimator, grid in grids:
        search = GridSearchCV(estimator, grid, cv=5, scoring="accuracy", n_jobs=1, error_score="raise", return_train_score=True)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        records.append({"model": name, "best_params": search.best_params_, "best_score": float(search.best_score_)})
    model = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1)), ("tree", fitted["tree"]), ("gradien", fitted["gradient"]), ("bayes", fitted["bayes"])],
        final_estimator=fitted["logistic"],
        cv=5,
        n_jobs=1,
    )
    return model, records


def _metrics(y_true, prediction, probability):
    result = classification_metrics(y_true, prediction, probability)
    result["precision"] = float(precision_score(y_true, prediction, zero_division=0))
    result["recall"] = float(recall_score(y_true, prediction, zero_division=0))
    return result


def _plots(y_true, prediction, probability, out: Path, label: str):
    out.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, probability)
    plt.savefig(out / f"{label}_roc.png", dpi=140)
    plt.close()
    PrecisionRecallDisplay.from_predictions(y_true, probability)
    plt.savefig(out / f"{label}_precision_recall.png", dpi=140)
    plt.close()
    ConfusionMatrixDisplay.from_predictions(y_true, prediction)
    plt.savefig(out / f"{label}_confusion_matrix.png", dpi=140)
    plt.close()


def run_one(dataset_path: Path, dataset_name: str, track: str, method: str, output_root: Path):
    X, y, target = _dataset(dataset_path, dataset_name)
    X_train, X_test, y_train, y_test = _split(X, y, track=track)
    if track == "corrected":
        fitted_preprocessor = _preprocessor(X)
        search_X = fitted_preprocessor.fit_transform(X_train, y_train)
        model, searches = (_ovec_search(search_X, y_train, track) if method == "ovec" else _osec_search(search_X, y_train))
        estimator = ImbPipeline([("preprocess", _preprocessor(X)), ("smote", SMOTE(random_state=SEED)), ("model", model)])
        fit_X, fit_y = X_train, y_train
    else:
        model, searches = (_ovec_search(X_train, y_train, track) if method == "ovec" else _osec_search(X_train, y_train))
        estimator = model
        fit_X, fit_y = X_train, y_train
    started = time.time()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit(fit_X, fit_y)
    probability = estimator.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    run_dir = output_root / dataset_name / track / method
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test.to_numpy(), "prediction": prediction, "probability": probability}).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame([_metrics(y_test, prediction, probability)]).to_json(run_dir / "metrics.json", orient="records", indent=2)
    pd.DataFrame(searches).to_json(run_dir / "search_results.json", orient="records", indent=2)
    joblib.dump(estimator, run_dir / "model.joblib")
    _plots(y_test, prediction, probability, run_dir / "figures", f"{dataset_name}_{track}_{method}")
    manifest = {
        "dataset": dataset_name, "dataset_path": str(dataset_path), "dataset_sha256": sha256_file(dataset_path),
        "target": target, "track": track, "method": method, "n_train": len(y_train), "n_test": len(y_test),
        "runtime_seconds": time.time() - started, "warnings": [str(item.message) for item in captured],
        "search_results": searches, "preprocessing": "training-only corrected pipeline" if track == "corrected" else "author source order: encoding/scaling, SMOTE, split",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    reloaded = joblib.load(run_dir / "model.joblib")
    if not np.array_equal(reloaded.predict(X_test), prediction):
        raise RuntimeError(f"Reloaded model predictions differ: {run_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", default=["five_states", "mendeley"])
    parser.add_argument("--tracks", nargs="+", default=["author_faithful", "compatibility_adapted", "corrected"])
    parser.add_argument("--methods", nargs="+", default=["ovec", "osec"])
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    paths = {
        "five_states": args.root / "reference/author_release/heart5.csv",
        "mendeley": args.root / "reference/author_release/MendeleyCardiovascular_Disease_Dataset.csv",
    }
    output = args.output_root or (args.root / "results/ensemble_experiments")
    manifests = []
    for dataset in args.datasets:
        for track in args.tracks:
            for method in args.methods:
                try:
                    manifests.append(run_one(paths[dataset], dataset, track, method, output))
                except (OSError, RuntimeError, ValueError) as exc:
                    failure_dir = output / dataset / track / method
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    failure = {
                        "dataset": dataset,
                        "track": track,
                        "method": method,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    (failure_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                    manifests.append(failure)
                    print(f"FAILED {dataset}/{track}/{method}: {type(exc).__name__}: {exc}")
    (output / "experiment_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
