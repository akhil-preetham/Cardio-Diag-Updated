# Execution blockers

No notebook execution was authorized or performed.

## Blockers applying to the collection
- Most notebooks require Google Colab imports and `/content/drive` paths that are unavailable in the local runner.
- Several notebooks install packages with `!pip`, write external artifacts, or contain deletion/shell operations.
- Specialized dependencies such as `loras`, `lime`, LightGBM, CatBoost, ReportLab, image tooling, and plotting extras are not part of the reviewed runner requirements.
- Referenced serialized models, PDFs, CSVs, and Colab Drive files require explicit existence and provenance checks.
- Historical saved outputs cannot be treated as reproduced results.

## Notebook-specific dependency and file blockers
### `base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb`
- Missing/unavailable imports: `google`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv`
- Potential API issues: none detected

### `five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb`
- Missing/unavailable imports: `google|loras|openpyxl|seaborn`
- External files requiring verification: ` - ALL_Subgroup_Performance.csv| - ALL_Subgroup_Performance_with_CI_Formatted.csv| - Overall_Performance.csv| - Overall_Performance_with_CI_Formatted.csv|/content/drive/MyDrive/heart5.csv|/content/drive/MyDrive/heart_dataset.csv|ALL_Subgroup_Performance.csv|ALL_Subgroup_Performance_with_CI_Formatted.csv|All_Subgroup_Performance_with_CI.csv|Overall_Performance.csv|Overall_Performance_with_CI.csv|Overall_Performance_with_CI_Formatted.csv|Subgroup_Fairness_Results.xlsx|forest_plot_{metric}.pdf|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb`
- Missing/unavailable imports: `google|loras|openpyxl|seaborn`
- External files requiring verification: ` - ALL_Subgroup_Performance.csv| - ALL_Subgroup_Performance_with_CI_Formatted.csv| - Overall_Performance.csv| - Overall_Performance_with_CI_Formatted.csv|/content/drive/MyDrive/heart5.csv|/content/drive/MyDrive/heart_dataset.csv|ALL_Subgroup_Performance.csv|ALL_Subgroup_Performance_with_CI_Formatted.csv|All_Subgroup_Performance_with_CI.csv|Overall_Performance.csv|Overall_Performance_with_CI.csv|Overall_Performance_with_CI_Formatted.csv|Subgroup_Fairness_Results.xlsx|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb`
- Missing/unavailable imports: `google|loras`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb`
- Missing/unavailable imports: `google|lightgbm|loras|seaborn`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb`
- Missing/unavailable imports: `google|lime|loras|seaborn`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv`
- Potential API issues: none detected

### `five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb`
- Missing/unavailable imports: `google|lime|loras|matplotlib_venn|seaborn`
- External files requiring verification: `/content/correlation_heatmap.pdf|/content/drive/MyDrive/heart5.csv|correlation_plot.pdf`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb`
- Missing/unavailable imports: `google|imgkit|lime|loras|reportlab|seaborn`
- External files requiring verification: `/content/LIME_directionality_plot.pdf|/content/NOVEM_CVD_All_Plots.pdf|/content/drive/MyDrive/heart5.csv|/content/drive/MyDrive/models/LoRAS_train_test_data.pkl|/content/drive/MyDrive/models/voting_pipeline.pkl|LIME_PCA_contributions.csv|LIME_PCA_summary.pdf|LIME_PCA_summary.png|LIME_PCA_summary_fixed.pdf|LIME_absolute_importance.pdf|LIME_absolute_importance.png|LIME_absolute_importance_{timestamp}.pdf|LIME_absolute_importance_{timestamp}.png|LIME_base_vs_ensemble.csv|LIME_base_vs_ensemble_overlap.pdf|LIME_base_vs_ensemble_stability.pdf|LIME_baseline_global_importance.csv|LIME_combined_comparison.pdf|LIME_combined_comparison.png|LIME_feature_importance.pdf|LIME_feature_importance.png|LIME_feature_importance_abs.png|LIME_feature_importance_directional.png|LIME_feature_importance_{timestamp}.png|LIME_jaccard_{model_name}.csv|LIME_multirun_jaccard.csv|LIME_multirun_stability.pdf|LIME_original_features_summary.pdf|LIME_original_features_summary.png|LIME_perturbation_sensitivity.csv|LIME_perturbation_sensitivity.pdf|LIME_report.pdf|LIME_report_{timestamp}.pdf|LIME_samplecount_sensitivity.csv|LIME_samplecount_sensitivity.pdf|LIME_signed.pdf|LIME_signed.png|LIME_signed_importance_{timestamp}.pdf|LIME_signed_importance_{timestamp}.png|LIME_stability_boxplot.pdf|LIME_stability_boxplot.png|LIME_stability_heatmap.pdf|LIME_stability_heatmap.png|LIME_stability_jaccard.png|LIME_stability_sd.png|LIME_stability_summary.pdf|LIME_stability_summary.png|LIME_summary.csv|LIME_summary_{timestamp}.csv|LoRAS_train_test_data.pkl|_small.png|heart_disease_distribution.pdf|lime_instances/lime_instance_{idx}.png|pca_feature_contributions.pdf|predictions.csv|raw_explanations.csv|roc_curve.pdf|voting_model_results.csv|voting_model_roc.png|voting_pipeline.pkl|✅ Heatmap saved as pca_feature_contributions.pdf|✅ Plot saved as heart_disease_distribution.pdf|✅ ROC Curve saved as roc_curve.pdf|✅ Saved LIME_signed.png|✅ Saved LIME_summary.csv|✅ Saved: LIME_base_vs_ensemble.csv|✅ Saved: LIME_feature_importance_abs.png|✅ Saved: LIME_feature_importance_directional.png|✅ Saved: LIME_perturbation_sensitivity.csv|✅ Saved: LIME_samplecount_sensitivity.csv|✅ Saved: LIME_stability_jaccard.png|✅ Saved: LIME_stability_sd.png`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb`
- Missing/unavailable imports: `google|lime|loras|matplotlib_venn|seaborn`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb`
- Missing/unavailable imports: `google|lime|loras|matplotlib_venn|seaborn`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb`
- Missing/unavailable imports: `google|lightgbm|loras|seaborn`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb`
- Missing/unavailable imports: `catboost|google|lightgbm|loras|seaborn`
- External files requiring verification: `/content/DCA_OVEC_300dpi.pdf|/content/calibration_curve_300dpi.pdf|/content/confusion_matrix_ys66.pdf|/content/drive/MyDrive/heart5.csv|/content/novec_confusion_matrix.pdf|/content/pr_curve_300dpi.pdf|/content/voting_confusion_matrix.pdf|ROC_NOVEC_NOSEC.pdf|calibration_curve.pdf|decision_curve.pdf|precision_recall_curve.pdf|roc_curve_plot.pdf|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb`
- Missing/unavailable imports: `google|loras|seaborn`
- External files requiring verification: `/content/drive/MyDrive/heart5.csv|ROC_NOVEC_NOSEC.pdf|roc_curve_plot.pdf|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb`
- Missing/unavailable imports: `google|imgkit|lime|loras|reportlab|seaborn`
- External files requiring verification: `/content/LIME_directionality_plot.pdf|/content/NOVEM_CVD_All_Plots.pdf|/content/drive/MyDrive/heart5.csv|/content/drive/MyDrive/models/LoRAS_train_test_data.pkl|/content/drive/MyDrive/models/voting_pipeline.pkl|LIME_PCA_contributions.csv|LIME_PCA_summary.pdf|LIME_PCA_summary.png|LIME_PCA_summary_fixed.pdf|LIME_absolute_importance.pdf|LIME_absolute_importance.png|LIME_absolute_importance_{timestamp}.pdf|LIME_absolute_importance_{timestamp}.png|LIME_base_vs_ensemble.csv|LIME_base_vs_ensemble_overlap.pdf|LIME_base_vs_ensemble_stability.pdf|LIME_baseline_global_importance.csv|LIME_combined_comparison.pdf|LIME_combined_comparison.png|LIME_feature_importance.pdf|LIME_feature_importance.png|LIME_feature_importance_abs.png|LIME_feature_importance_directional.png|LIME_feature_importance_{timestamp}.png|LIME_jaccard_{model_name}.csv|LIME_multirun_jaccard.csv|LIME_multirun_stability.pdf|LIME_original_features_summary.pdf|LIME_original_features_summary.png|LIME_perturbation_sensitivity.csv|LIME_perturbation_sensitivity.pdf|LIME_report.pdf|LIME_report_{timestamp}.pdf|LIME_samplecount_sensitivity.csv|LIME_samplecount_sensitivity.pdf|LIME_signed.pdf|LIME_signed.png|LIME_signed_importance_{timestamp}.pdf|LIME_signed_importance_{timestamp}.png|LIME_stability_boxplot.pdf|LIME_stability_boxplot.png|LIME_stability_heatmap.pdf|LIME_stability_heatmap.png|LIME_stability_jaccard.png|LIME_stability_sd.png|LIME_stability_summary.pdf|LIME_stability_summary.png|LIME_summary.csv|LIME_summary_{timestamp}.csv|LoRAS_train_test_data.pkl|_small.png|heart_disease_distribution.pdf|lime_instances/lime_instance_{idx}.png|pca_feature_contributions.pdf|predictions.csv|raw_explanations.csv|roc_curve.pdf|voting_model_results.csv|voting_model_roc.png|voting_pipeline.pkl|✅ Heatmap saved as pca_feature_contributions.pdf|✅ Plot saved as heart_disease_distribution.pdf|✅ ROC Curve saved as roc_curve.pdf|✅ Saved LIME_signed.png|✅ Saved LIME_summary.csv|✅ Saved: LIME_base_vs_ensemble.csv|✅ Saved: LIME_feature_importance_abs.png|✅ Saved: LIME_feature_importance_directional.png|✅ Saved: LIME_perturbation_sensitivity.csv|✅ Saved: LIME_samplecount_sensitivity.csv|✅ Saved: LIME_stability_jaccard.png|✅ Saved: LIME_stability_sd.png`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb`
- Missing/unavailable imports: `google|loras|openpyxl|seaborn`
- External files requiring verification: ` - ALL_Subgroup_Performance.csv| - ALL_Subgroup_Performance_with_CI_Formatted.csv| - Overall_Performance.csv| - Overall_Performance_with_CI_Formatted.csv|/content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv|/content/drive/MyDrive/heart_mendeleyset.csv|ALL_Subgroup_Performance.csv|ALL_Subgroup_Performance_with_CI_Formatted.csv|All_Subgroup_Performance_with_CI.csv|Overall_Performance.csv|Overall_Performance_with_CI.csv|Overall_Performance_with_CI_Formatted.csv|Subgroup_Fairness_Results.xlsx|forest_plot_{metric}.pdf|your_mendeleyset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb`
- Missing/unavailable imports: `google|loras|openpyxl|seaborn`
- External files requiring verification: ` - ALL_Subgroup_Performance.csv| - ALL_Subgroup_Performance_with_CI_Formatted.csv| - Overall_Performance.csv| - Overall_Performance_with_CI_Formatted.csv|/content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv|/content/drive/MyDrive/heart_mendeleyset.csv|ALL_Subgroup_Performance.csv|ALL_Subgroup_Performance_with_CI_Formatted.csv|All_Subgroup_Performance_with_CI.csv|Overall_Performance.csv|Overall_Performance_with_CI.csv|Overall_Performance_with_CI_Formatted.csv|Subgroup_Fairness_Results.xlsx|forest_plot_{metric}.pdf|your_mendeleyset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb`
- Missing/unavailable imports: `google|loras|seaborn`
- External files requiring verification: `/content/DCA_OVEC_300dpi.pdf|/content/calibration_curve_300dpi.pdf|/content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv|/content/drive/MyDrive/heart5.csv|/content/pr_curve_300dpi.pdf|NOSEC_Model_Comparison.pdf|OVEC_calibration_curve.pdf|OVEC_calibration_curve_300dpi.pdf|calibration_quantile_300dpi.pdf|uniform_calibration_diagnostic_300dpi.pdf|your_dataset.csv|your_mendeleyset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

### `mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb`
- Missing/unavailable imports: `google|lightgbm|loras|seaborn`
- External files requiring verification: `/content/drive/MyDrive/MendeleyCardiovascular_Disease_Dataset.csv|/content/drive/MyDrive/heart5.csv|your_dataset.csv`
- Potential API issues: DataFrame.append may be removed in current pandas.

## Required gate before any execution
1. Approve an exact notebook allowlist and execution order.
2. Create and hash working copies; never edit the originals.
3. Resolve every Colab path to a verified local input or mark the notebook non-reproducible.
4. Review every shell, package-install, network, deletion, and file-write operation.
5. Capture package versions and run in an isolated output directory.