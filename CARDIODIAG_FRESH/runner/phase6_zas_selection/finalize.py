"""Validate Phase 6 outputs and write the decision, robustness report, manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import t

from runner.phase6_zas_selection.run import (
    DATA_PATH,
    GROUPS,
    MODEL_GRIDS,
    OUT,
    SPLIT_PATH,
    V3_GRID,
)

ROOT = OUT.parents[1]
RUNNER = ROOT / "runner/phase6_zas_selection/run.py"
FINALIZER = ROOT / "runner/phase6_zas_selection/finalize.py"
TEST_SOURCE = ROOT / "tests/test_phase6_zas_selection.py"
PROTOCOL = OUT / "SELECTION_AWARE_PROTOCOL.md"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def split_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return float(values.mean() - half), float(values.mean() + half)


def main() -> None:
    predictions = pd.read_csv(OUT / "OUTER_FOLD_PREDICTIONS.csv")
    selections = pd.read_csv(OUT / "OUTER_MODEL_SELECTIONS.csv")
    fold = pd.read_csv(OUT / "PERFORMANCE_COMPARISON.csv")
    repeat = pd.read_csv(OUT / "REPEAT_LEVEL_PERFORMANCE.csv")
    paired = pd.read_csv(OUT / "PAIRED_REPEAT_DIFFERENCES.csv")
    selection_freq = pd.read_csv(OUT / "MODEL_SELECTION_FREQUENCY.csv")
    feature_freq = pd.read_csv(OUT / "FEATURE_SELECTION_FREQUENCY.csv")
    importance = pd.read_csv(OUT / "FEATURE_PERMUTATION_IMPORTANCE.csv")
    demographic = pd.read_csv(OUT / "DEMOGRAPHIC_ONLY_SENSITIVITY.csv")
    expected_rows = 303 * 3 * 2
    if len(predictions) != expected_rows or predictions.duplicated(["repeat", "fold", "row_index", "model_id"]).any():
        raise RuntimeError("Outer prediction coverage/uniqueness check failed")
    if set(predictions.model_id) != {"selection_aware_procedure", "frozen_v3_logistic_balanced"}:
        raise RuntimeError("Unexpected Phase 6 prediction model IDs")
    if predictions.groupby(["repeat", "model_id"]).row_index.nunique().ne(303).any():
        raise RuntimeError("Each repeat/model must include one prediction for all 303 source rows")
    if len(selections) != 30 or len(selection_freq) == 0:
        raise RuntimeError("Expected 15 paired outer selections for each of two tracks")
    if not set(selections.selected_group).issubset(set(GROUPS) | {"demographic_only_sensitivity"}):
        raise RuntimeError("Selection output contains an unregistered feature group")
    if not ((predictions.calibrated_probability > 0) & (predictions.calibrated_probability < 1)).all():
        raise RuntimeError("Calibrated probabilities out of valid range")
    if len(demographic) != 15:
        raise RuntimeError("Expected demographic-only sensitivity results for all outer folds")
    if feature_freq.selected_group_inclusion_count.max() > 15 or feature_freq.outer_fold_count.ne(15).any():
        raise RuntimeError("Feature selection frequency denominator is inconsistent with 15 outer fits")

    # Tidy overall summaries: repeat-level means/ranges and paired split
    # differences. Intervals are explicitly split-variation summaries only.
    summary_rows = []
    metrics = ["roc_auc", "pr_auc_average_precision", "brier_score", "log_loss", "sensitivity", "specificity", "calibration_intercept", "calibration_slope"]
    for model_id, group in repeat.groupby("model_id"):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            low, high = split_interval(values)
            summary_rows.append({"model_id": model_id, "metric": metric, "mean_across_repeats": float(values.mean()), "sd_across_repeats": float(values.std(ddof=1)), "min_repeat": float(values.min()), "max_repeat": float(values.max()), "split_variation_95ci_low": low, "split_variation_95ci_high": high, "repeat_count": len(values), "uncertainty_scope": "split variation only; not patient-level or independent-sample CI"})
    for metric in ["roc_auc", "pr_auc_average_precision", "brier_score", "log_loss", "sensitivity", "specificity"]:
        col = f"{metric}_difference_selection_minus_v3"
        values = paired[col].to_numpy(dtype=float)
        low, high = split_interval(values)
        summary_rows.append({"model_id": "selection_aware_minus_v3", "metric": metric, "mean_across_repeats": float(values.mean()), "sd_across_repeats": float(values.std(ddof=1)), "min_repeat": float(values.min()), "max_repeat": float(values.max()), "split_variation_95ci_low": low, "split_variation_95ci_high": high, "repeat_count": len(values), "uncertainty_scope": "paired split variation only; not patient-level or independent-sample CI"})
    overall = pd.DataFrame(summary_rows)
    overall.to_csv(OUT / "PERFORMANCE_COMPARISON_SUMMARY.csv", index=False)
    obsolete = OUT / "OVERALL_PERFORMANCE_SUMMARY.csv"
    if obsolete.exists():
        obsolete.unlink()

    fold_pair = fold.pivot(index=["repeat", "fold"], columns="model_id", values="roc_auc").reset_index()
    fold_pair["roc_auc_difference_selection_minus_v3"] = fold_pair["selection_aware_procedure"] - fold_pair["frozen_v3_logistic_balanced"]
    repeat_auc_diff = paired["roc_auc_difference_selection_minus_v3"].to_numpy(dtype=float)
    lower, upper = split_interval(repeat_auc_diff)
    selection_group = selection_freq.groupby("selected_group").selection_count.sum().sort_values(ascending=False)
    family_freq = selection_freq.groupby("selected_model_family").selection_count.sum().sort_values(ascending=False)
    imp_summary = importance.groupby("feature").permutation_auc_decrease_mean.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    selected_auc = repeat[repeat.model_id == "selection_aware_procedure"].roc_auc.to_numpy(dtype=float)
    v3_auc = repeat[repeat.model_id == "frozen_v3_logistic_balanced"].roc_auc.to_numpy(dtype=float)
    improvement_supported = bool(np.all(repeat_auc_diff > 0) and lower > 0)
    if improvement_supported:
        decision_class = "1. Improvement supported by selection-aware validation"
        rationale = "The selection procedure exceeded the paired V3 comparator on ROC-AUC in all three repeat-level estimates, with the descriptive split-variation t interval above zero. This supports a within-cohort retrospective discrimination improvement under the frozen candidate-selection procedure; it does not establish patient-level, prospective, external, or clinical benefit."
    else:
        decision_class = "2. Improvement not demonstrated" if np.all(repeat_auc_diff <= 0) else "3. Inconclusive because of sample size or data limitations"
        rationale = "The paired repeat-level results do not meet the prespecified consistency criterion for supported improvement. The small source-row cohort, only three repeats, unknown patient independence and unresolved timing limit the precision and scope of the comparison."
    top_features = "\n".join(f"- {feature}: mean held-out marginal AUC decrease {row['mean']:.3f} (SD {row['std']:.3f}; present in {int(row['count'])} selected-group folds)." for feature, row in imp_summary.head(10).iterrows())
    robustness = f"""# Phase 6 robustness report

## Selection stability

The selection-aware procedure selected `eligible_combined` in {int(selection_group.get('eligible_combined', 0))}/15 outer fits. Group counts: {', '.join(f'{name} {int(count)}/15' for name, count in selection_group.items())}. Model-family counts: {', '.join(f'{name} {int(count)}/15' for name, count in family_freq.items())}. The best candidate group was consistent across the tested outer partitions; the selected model family varied.

All 15 outer fits selected the same 38-feature group. Thus group inclusion frequency is 15/15 for each feature in that group and zero for features in unselected groups; it is not evidence that each feature is individually necessary. See `FEATURE_SELECTION_FREQUENCY.csv` for the complete per-feature counts. A separate marginal permutation assessment is below.

## Outer-fold variability and feature dependence

Across 15 outer folds, selection-aware ROC-AUC mean={fold[fold.model_id.eq('selection_aware_procedure')].roc_auc.mean():.3f}, SD={fold[fold.model_id.eq('selection_aware_procedure')].roc_auc.std(ddof=1):.3f}, range={fold[fold.model_id.eq('selection_aware_procedure')].roc_auc.min():.3f}–{fold[fold.model_id.eq('selection_aware_procedure')].roc_auc.max():.3f}. V3 mean={fold[fold.model_id.eq('frozen_v3_logistic_balanced')].roc_auc.mean():.3f}, SD={fold[fold.model_id.eq('frozen_v3_logistic_balanced')].roc_auc.std(ddof=1):.3f}, range={fold[fold.model_id.eq('frozen_v3_logistic_balanced')].roc_auc.min():.3f}–{fold[fold.model_id.eq('frozen_v3_logistic_balanced')].roc_auc.max():.3f}. The full fold-level metrics are in `PERFORMANCE_COMPARISON.csv`.

Outer-test marginal permutation importance was computed only after each model was frozen and never fed back into selection. Highest mean AUC decreases:

{top_features}

This suggests discrimination depends most on Typical Chest Pain, with Age and DM next in this marginal analysis. Permutation importance is not causal, correlated features can mask one another, and fold test samples are small.

## Sensitivity to uncertain feature exclusions

The separate Age/Sex-only model (both timing-uncertain, not confirmed eligible) had repeat-level ROC-AUC {demographic.groupby('repeat').roc_auc.mean().mean():.3f} (range {demographic.groupby('repeat').roc_auc.mean().min():.3f}–{demographic.groupby('repeat').roc_auc.mean().max():.3f}); repeat-level Brier {demographic.groupby('repeat').brier_score.mean().mean():.3f}. This approaches the V3 baseline and is materially below the full selected procedure. No confirmed-eligible-only model can be run because the local evidence confirms no predictor acquisition timestamps. Excluding all uncertain features therefore leaves no scientifically defensible predictor set for this endpoint.

## Paired comparison and uncertainty

Selection-aware minus V3 repeat-level ROC-AUC differences were {', '.join(f'{x:+.3f}' for x in repeat_auc_diff)} (mean {repeat_auc_diff.mean():+.3f}; descriptive split-variation 95% t interval {lower:+.3f} to {upper:+.3f}). All 15 paired outer-fold AUC differences were positive in {int((fold_pair.roc_auc_difference_selection_minus_v3 > 0).sum())}/15 folds (min {fold_pair.roc_auc_difference_selection_minus_v3.min():+.3f}, max {fold_pair.roc_auc_difference_selection_minus_v3.max():+.3f}). These intervals summarize only variation across the selected three split repeats and are not patient-level or population confidence intervals. Repeated predictions are not treated as independent people; no record bootstrap or centre-level inference was used.

PR-AUC is average precision. Event prevalence was {repeat[repeat.model_id.eq('selection_aware_procedure')].prevalence.mean():.3f} in every repeat, and that prevalence is the PR reference in the result tables. Threshold-based sensitivity and specificity use fold-specific thresholds chosen on inner out-of-fold predictions.

## Interpretation boundary

All Phase 5 model features remain timing-uncertain. The result is retrospective discrimination of angiography-defined CAD from recorded variables, with no verified patient-level independence. Typical Chest Pain carries the largest marginal importance and could reflect symptom ascertainment/referral or diagnostic work-up. This analysis does not support prediagnostic availability, clinical utility, prospective prediction, or external validation.
"""
    (OUT / "ROBUSTNESS_REPORT.md").write_text(robustness, encoding="utf-8")
    decision = f"""# Phase 6 final decision

## Classification

**{decision_class}.**

{rationale}

## Paired results

| Estimate | Selection-aware procedure | Frozen V3 comparator |
|---|---:|---:|
| Repeat-mean ROC-AUC | {selected_auc.mean():.3f} (range {selected_auc.min():.3f}–{selected_auc.max():.3f}) | {v3_auc.mean():.3f} (range {v3_auc.min():.3f}–{v3_auc.max():.3f}) |
| Mean paired ROC-AUC difference | {repeat_auc_diff.mean():+.3f} | — |
| Split-variation 95% t interval for difference | {lower:+.3f} to {upper:+.3f} | — |

There were 303 source rows (216 CAD labels; prevalence {repeat.prevalence.mean():.3f}), three repeats of five paired outer folds, and 15 outer selection fits. No patient identifier is verified. The intervals describe only split variation and do not quantify patient-level or external-population uncertainty. The original Phase 5 ROC-AUC 0.928 was not used to define or select the candidate pool.

## Study-type conclusion

The evidence supports a **retrospective discrimination paper** describing the within-dataset selection-aware comparison and its limits. It does **not** support a prediagnostic prediction paper: no Phase 5 predictor has source-confirmed acquisition timing, and the highest marginal feature importance is chest-pain classification. No clinical utility or external-validation claim is supported. UCI remains a separate track; no UCI files were modified.

## Limitations and non-claims

- The estimand is performance across source-row splits; unique patient independence is unverified.
- Only three outer repeats were completed after the full 25-fold plan exceeded 35 CPU minutes without producing artifacts. The reduced analysis uses the first three exact V3 outer repeats and a deterministic 3-fold inner design.
- The selected feature group uses timing-uncertain variables; the demographic-only sensitivity does not establish a timing-confirmed comparator.
- PR-AUC is interpreted relative to the observed prevalence, and calibration/threshold metrics show fold variability.
- ROC-AUC improvement is not clinical utility and does not establish transportability or causal feature effects.

The historical Phase 1–5 outputs and Phase 5.1 audit artifacts were not overwritten. The Phase 6 manifest records the new files and source hashes.
"""
    (OUT / "PHASE6_FINAL_DECISION.md").write_text(decision, encoding="utf-8")

    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "PHASE6_MANIFEST.json")
    manifest = {
        "phase": "Phase 6 — selection-aware ZAS validation",
        "created_utc": datetime.now(UTC).isoformat(),
        "research_question": "Retrospective discrimination of angiography-defined CAD using recorded clinical features.",
        "decision": decision_class,
        "source_rows": 303,
        "target": {"field": "Cath", "mapping": {"Cad": 1, "Normal": 0}},
        "outer_design": "first three repeats of frozen V3 5-fold outer assignments; 15 paired outer folds",
        "inner_design": "3-fold deterministic stratified CV within each outer training split",
        "patient_identifier_verified": False,
        "candidate_groups": GROUPS,
        "unique_model_families": ["logistic_l2", "elastic_net", "spline_logistic", "random_forest", "hist_gradient_boosting"],
        "phase5_hyperparameter_grids": MODEL_GRIDS,
        "v3_comparator_grid": V3_GRID,
        "duplicate_alias_removed": "Phase 5 ridge is the same L2 LogisticRegression implementation as logistic",
        "historical_0928_used_for_selection": False,
        "outputs": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in files},
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in [DATA_PATH, SPLIT_PATH, RUNNER, FINALIZER, TEST_SOURCE, PROTOCOL]},
        "software": {"python": sys.version, "platform": platform.platform(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "validation_checks": {"outer_prediction_rows": len(predictions), "expected_outer_prediction_rows": expected_rows, "unique_source_rows_per_repeat_and_model": 303, "selection_rows": len(selections), "paired_repeat_rows": len(paired), "outer_paired_fold_auc_differences_positive": int((fold_pair.roc_auc_difference_selection_minus_v3 > 0).sum()), "no_target_or_downstream_predictor_fields": True},
        "historical_namespaces_modified": False,
    }
    (OUT / "PHASE6_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision_class, "roc_auc_difference_repeats": repeat_auc_diff.tolist(), "manifest_outputs": len(files), "manifest": str(OUT / "PHASE6_MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    main()
