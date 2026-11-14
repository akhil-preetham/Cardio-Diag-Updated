# CARDIODIAG reproducibility report

Generated: `2026-11-12T17:54:53+00:00`

## Evidence boundary

The author release was verified at commit `e63b02c8a2aba6344a48025f657696fd99ac3a5f` and was clean at preflight. Original author files and extracted notebooks were not modified. Static audit covered 18 notebooks without executing them.

## Author-faithful execution

The explicitly allowlisted base notebook was copied and attempted. It failed before producing a reproduced result because it depends on the historical `google.colab`/`/content/drive` environment. The remaining notebooks were not allowlisted because static audit found Colab mounts, shell commands, deletion operations, or unresolved external artifacts.

- `base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb`: **failed**
- `five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb`: **skipped_not_allowlisted**
- `five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb`: **skipped_not_allowlisted**
- `lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb`: **skipped_not_allowlisted**
- `mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb`: **skipped_not_allowlisted**
- `mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb`: **skipped_not_allowlisted**
- `mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb`: **skipped_not_allowlisted**
- `mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb`: **skipped_not_allowlisted**

Historical saved values in the original notebook (accuracy 0.8824, precision 0.8898, recall 0.9052, F1 0.8974, ROC-AUC 0.9578) are treated as reported artifacts, not independently reproduced results.

## Corrected validation

The independent comparator uses a stratified 75:25 split, five-fold stratified CV within the training partition, training-fold preprocessing, SMOTE inside the imbalanced-learn pipeline, and ten seeded repeated holdouts. Mendeley `patientid` was excluded. Repeated holdouts overlap and therefore are not independent external validation cohorts.

See `experiment_results.csv` and the per-dataset JSON reports for every run and metric. The high Mendeley performance is an observed result on the supplied dataset, not evidence of external clinical validity.

## Conclusions

- The datasets and environment are reproducible from the preserved local artifacts.
- The historical base-paper values were not verified by a completed author-faithful execution.
- The corrected comparator supports the leakage-safe split/resampling protocol implemented here.
- Specialized claims (LoRAS, XAIH-DRC, OVEC, OSEC, LIME, fairness, clinical utility, calibration, statistical tests, and external validation) remain unverified in this run.
