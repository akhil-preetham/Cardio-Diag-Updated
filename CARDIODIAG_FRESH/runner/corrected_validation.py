from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from common import load_config, resolve, utc_now, write_json
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main(config_file, dataset_override=None):
    cfg, root = load_config(Path(config_file).resolve())
    settings = cfg["corrected_validation"]
    if not settings.get("enabled", False):
        print("Corrected validation is disabled in config. Nothing executed.")
        return

    ds_name = dataset_override or settings["dataset"]
    if ds_name not in cfg["datasets"]:
        raise SystemExit(f"Unknown dataset key: {ds_name}")
    data_path = resolve(root, cfg["datasets"][ds_name])
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    target = "HeartDisease" if ds_name == "five_states" else "target"
    if target not in df.columns:
        raise SystemExit(f"Target column {target} not found.")

    y = pd.to_numeric(df[target], errors="raise").astype(int)
    X = df.drop(columns=[target]).copy()
    identifier_columns = [
        c for c in X.columns
        if c.lower() in {"id", "patientid", "patient_id", "subject_id"} or c.lower().endswith("_id")
    ]
    X = X.drop(columns=identifier_columns)

    cat_cols = [c for c in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].astype("string")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SkPipeline([("impute", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler())]), num_cols),
            ("cat", SkPipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ],
        remainder="drop", sparse_threshold=0.0
    )

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                      random_state=42, n_jobs=1)),
        ("tree", DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                          max_depth=2, random_state=42)),
        ("gnb", GaussianNB())
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5, n_jobs=1
    )
    pipe = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", stack)
    ])

    repeats = int(settings.get("repeats", 10))
    test_size = float(settings.get("test_size", 0.25))
    folds = int(settings.get("cv_folds", 5))
    seed = int(settings.get("random_seed", 42))
    rows = []
    out_root = resolve(root, cfg["output_root"])
    results_dir = out_root / "corrected_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    for i in range(repeats):
        run_seed = seed + i
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=run_seed
        )
        # SMOTE is inside the imbalanced-learn pipeline, so it is fitted only
        # on each training fold during CV and on the full training partition at refit.
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=run_seed)
        search = GridSearchCV(
            pipe, param_grid={"model__final_estimator__C": [0.1, 1.0, 10.0]},
            scoring="roc_auc", cv=cv, n_jobs=int(settings.get("n_jobs", 1)),
            refit=True, error_score="raise", return_train_score=False
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        prob = best.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        row = {
            "run": i + 1, "seed": run_seed, "n_train": len(X_train), "n_test": len(X_test),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall_sensitivity": recall_score(y_test, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else None,
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc_average_precision": average_precision_score(y_test, prob),
            "brier_score": brier_score_loss(y_test, prob),
            "best_cv_roc_auc": search.best_score_,
            "best_params": search.best_params_,
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]
        }
        prediction_frame = pd.DataFrame({
            "run": i + 1,
            "seed": run_seed,
            "row_index": y_test.index,
            "y_true": y_test.to_numpy(),
            "probability": prob,
            "prediction": pred,
        })
        prediction_frame.to_csv(results_dir / f"{ds_name}_predictions_run_{i + 1:02d}.csv", index=False)
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_dir / f"{ds_name}_corrected_runs.csv", index=False)
        print(f"Run {i+1}/{repeats}: ROC-AUC={row['roc_auc']:.4f}, accuracy={row['accuracy']:.4f}")

    summary = {}
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity",
                   "specificity", "f1", "roc_auc", "pr_auc_average_precision", "brier_score"]:
        vals = pd.Series([r[metric] for r in rows], dtype="float64")
        summary[metric] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                           "min": float(vals.min()), "max": float(vals.max())}
    report = {
        "created_utc": utc_now(),
        "track": "corrected_validation",
        "dataset": str(data_path),
        "n_rows": len(df), "n_features": X.shape[1], "target": target,
        "excluded_identifier_columns": identifier_columns,
        "protocol": {
            "repeated_stratified_holdout_runs": repeats, "test_size": test_size,
            "cv_folds": folds, "selection_metric": "ROC-AUC",
            "smote_location": "inside imblearn pipeline; training folds only",
            "categorical_encoding": "OneHotEncoder",
            "numeric_preprocessing": "median imputation + standardization",
            "test_evaluation": "one evaluation per run after model selection",
            "identifier_policy": "exclude columns recognized as patient identifiers",
            "important_caveat": "Repeated holdouts overlap and are not independent external validation cohorts. This is a corrected comparator, not an exact reproduction of the paper."
        },
        "summary": summary,
        "runs": rows
    }
    write_json(results_dir / f"{ds_name}_corrected_validation_report.json", report)
    print(f"Corrected validation outputs: {results_dir}")

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python runner/corrected_validation.py config/runner_config.json [dataset_key]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
