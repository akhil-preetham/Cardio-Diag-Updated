# Methodology compliance

| Paper/source component | Repository evidence | Classification |
|---|---|---|
| Data preprocessing and categorical encoding | Extracted notebooks; `runner/ensemble_experiments.py` | Faithfully represented in author track; corrected track intentionally adapts to fold-local preprocessing. |
| SMOTE | Author-faithful notebook and `_split`; corrected pipeline | Author-faithful execution preserves pre-split behavior; corrected implementation is a methodological adaptation. |
| LoRAS | `src/cardiodiag/imbalance/sampling.py`, `results/data_integrity_and_loras/` | Executed comparison; exact paper-level equivalence requires parameter/source verification. |
| XAIH-DRC | `src/cardiodiag/feature_selection/xaih_drc.py`, `results/xaih_drc_experiments/` | Recovered/implemented procedure; exact paper equivalence not established. |
| OVEC | `src/cardiodiag/ensembles/classifiers.py`, ensemble manifests | Source-shaped implementation; author-faithful candidate set fails on current sklearn. Compatibility result is separate. |
| OSEC | `src/cardiodiag/ensembles/classifiers.py`, ensemble manifests | Source-shaped stacking implementation executed in saved tracks; protocol differs between tracks. |
| LIME | `src/cardiodiag/explainability/lime_workflow.py`, `results/lime_experiments/` | Artifacts exist; independent model/input provenance is incomplete. |
| Evaluation | `results/experiment_metrics.json`, corrected reports | Executed metrics are traceable for listed artifacts; not all paper metrics are recorded. |

## Historical reproduction

The prepared notebook execution demonstrates that a local compatibility copy can run. It does not establish historical environment equivalence. The unadapted copy is blocked by its Colab mount, and the author-faithful OVEC controlled path is blocked by an invalid current estimator parameter. These failures are preserved rather than silently repaired.

## Protocol distinctions

The author-faithful result reports the source behavior, including pre-split SMOTE and test-set-driven stopping. The corrected comparator uses stratified repeated holdouts, training-fold preprocessing and SMOTE, and training-only model selection. Because multiple protocol components differ, the corrected-versus-author-faithful metric difference is not a causal estimate of leakage bias.

## Required correction before publication

Use separate tables and labels for `author_faithful`, `compatibility_adapted`, and `corrected`. Do not describe the corrected comparator as an exact reproduction, and do not describe the repeated holdouts as external validation.
