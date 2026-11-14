# Claims and limitations

## Supported

- The Phase 5 ZAS 0.928214 AUC is reconstructible but exploratory because it was chosen after inspecting 36 reported group/model rows.
- Phase 6 provides a selection-aware, within-source-row comparison procedure. Its three repeat-level AUCs exceeded its paired V3 comparator; Phase 6.1 reconstructed the saved partitions, predictions and metrics.
- UCI Phase 4.3 supplies four-centre held-out, staged cohort evidence for its own `num` endpoint and tasks.
- Phase 6.1 confirms calibration selection dependence and that the “four folds” selection metadata was a reporting-only error for an actual three-fold process.

## Required limitations

- **Study framing:** retrospective discrimination of angiography-defined CAD using recorded clinical features.
- **Identity:** all ZAS split/evaluation units are source rows; unique patient identity and patient-level independence are unestablished. UCI intervals are record-level and do not support patient- or centre-level uncertainty.
- **Timing:** no predictor has source-confirmed acquisition timing. Treat every ZAS predictor in V3/Phase 6 as timing uncertain. Thus Phase 6 cannot establish prediagnostic prediction.
- **Calibration:** the selected hyperparameters/group use outer-training labels before calibration cross-fitting, making the calibration OOF scores selection-dependent. Training sizes differ between calibration folds and final refit. Report calibration as limited.
- **Selection:** Phase 6 assesses a finite frozen candidate pool selected inside outer training; only three outer repeats were completed. The split-variation interval is not population inference.
- **Feature interpretation:** group inclusion counts are not feature necessity. Marginal permutation importance is descriptive, potentially affected by correlated features, and not causal.
- **Prevalence:** interpret PR-AUC/AP against observed prevalence (ZAS 0.712871; UCI cohort prevalence differs). Do not compare AP across endpoints/cohorts as though directly comparable.
- **Utility:** higher AUC does not establish clinical utility, net benefit, safe thresholding or readiness.
- **External validation:** no independent compatible ZAS cohort was evaluated. UCI must remain a separate track.
- **UCI task separation:** Stage 2 represents an intended early-presentation information set; Stage 4 includes exercise-test information and is a different later task. Stage design labels do not prove timestamps.
- **Reproducibility:** project has no Git metadata; current working-tree status is unavailable. Existing paper draft predates Phase 6 and has unconfirmed author/ethics/funding declarations and unrendered LaTeX layout.

## Avoid

Do not say “prospective prediction,” “available before diagnosis,” “externally validated,” “patient-level confidence interval,” “clinically useful,” “safe,” “ready for deployment,” or “V3 external validation.” Do not present Phase 5 0.928 as an unbiased final AUC. Do not call Phase 6 calibration independently validated.
