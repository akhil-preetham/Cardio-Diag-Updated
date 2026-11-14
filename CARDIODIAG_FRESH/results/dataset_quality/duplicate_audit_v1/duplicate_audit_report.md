# Duplicate record audit

Generated UTC: `2026-11-13T06:53:12.324989+00:00`

## Scope and method

This diagnostic audit examines only duplicate records and their effect on partition overlap. Source CSVs are read without modification. Row indices are zero-based indices after CSV loading (the CSV header is not counted). Exact duplicates are grouped by all columns; predictor-only groups exclude the target, and Mendeley analyses also report the identifier policy separately.

Near-duplicate candidates use a conservative deterministic screen: all categorical predictors must match, every numeric predictor must be non-missing and differ by no more than 1% of that column's observed range, and candidates are not merged or treated as confirmed duplicates. The candidate-generation block is the categorical signature, avoiding unrestricted comparisons across dissimilar records.

Partition overlap reproduces the corrected-validation holdout protocol (stratified 75/25 train/test, seeds 42–51). It checks overlap within each train/test partition only; repeated holdouts naturally reuse observations across different runs.

## heart5

- Source: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\author_release\heart5.csv`
- Shape: 918 rows × 12 columns; target `HeartDisease`
- Exact duplicate groups: 0; subsequent duplicate occurrences: 0 (0.000%)
- Exact groups with conflicting labels: 0
- Predictor-only duplicate groups: 0; conflicting-label groups: 0
- Near-duplicate candidate pairs: 0; connected candidate groups: 0; conflicting-label pairs: 0

### Interpretation

Duplicate groups are diagnostic evidence, not proof of leakage. Cross-partition leakage requires the same record or predictor profile to appear in both train and test.

### Diagnostic scenarios

- original_dataset: 918 rows; class distribution {"0": 410, "1": 508}
- exact_duplicate_rows_excluded: 918 rows; class distribution {"0": 410, "1": 508}
- identical_predictor_groups_conservatively_deduplicated: 918 rows; class distribution {"0": 410, "1": 508}
- conflicting_label_groups_flagged_not_removed: 918 rows; class distribution {"0": 410, "1": 508}

### Partition overlap

- Affected holdouts with full-row overlap: 0/10; predictor-only overlap: 0/10; conflicting-label overlap: 0/10.

## mendeley

- Source: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\author_release\MendeleyCardiovascular_Disease_Dataset.csv`
- Shape: 1000 rows × 14 columns; target `target`
- Exact duplicate groups: 0; subsequent duplicate occurrences: 0 (0.000%)
- Exact groups with conflicting labels: 0
- Predictor-only duplicate groups: 0; conflicting-label groups: 0
- Near-duplicate candidate pairs: 0; connected candidate groups: 0; conflicting-label pairs: 0

### Interpretation

Duplicate groups are diagnostic evidence, not proof of leakage. Cross-partition leakage requires the same record or predictor profile to appear in both train and test.
- `patientid` is excluded by the corrected-validation implementation. Repeated patient IDs: 0; IDs with conflicting observations: 0.

### Diagnostic scenarios

- original_dataset: 1000 rows; class distribution {"0": 420, "1": 580}
- exact_duplicate_rows_excluded: 1000 rows; class distribution {"0": 420, "1": 580}
- identical_predictor_groups_conservatively_deduplicated: 1000 rows; class distribution {"0": 420, "1": 580}
- conflicting_label_groups_flagged_not_removed: 1000 rows; class distribution {"0": 420, "1": 580}

### Partition overlap

- Affected holdouts with full-row overlap: 0/10; predictor-only overlap: 0/10; conflicting-label overlap: 0/10.

## Conclusions

No record was removed from a source or used to produce model results. Exact duplicate rows with identical labels can inflate apparent sample size, while conflicting labels require separate review. Identical predictors do not establish patient identity, and near-duplicate candidates are not confirmed duplicates.

The unusually high Mendeley performance can be attributed to duplicate leakage only if the partition table shows train/test overlap under the same protocol. If it does not, duplicates alone do not explain that performance; dataset separability and other design/data issues remain outside this Problem 1 audit.

Limitations: no patient-level ground truth is available beyond the supplied identifier; the near-duplicate threshold is a conservative screening rule; repeated holdouts are not independent external cohorts; and the approximately 70,000-row Kaggle dataset was not available locally.
