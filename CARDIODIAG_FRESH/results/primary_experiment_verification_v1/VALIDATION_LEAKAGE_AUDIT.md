# Validation and Leakage Audit

## Classification

- **Verified correct:** 5 repeats, 5 outer folds per repeat, 4 inner folds, stratified outer and inner splitters, deterministic seeds, and zero recorded outer train/test overlaps.
- **Verified correct:** each row appears in exactly one outer test fold per repeat; each model has complete outer-test predictions.
- **Correct with limitations:** the source pipeline fits preprocessing inside each model pipeline and performs hyperparameter selection on outer-training data only. Inner assignments are reproducible from the source seed formula but were not saved as a separate artifact.
- **Unresolved:** patient-level grouping cannot be independently established from the derivative because no patient identifier is present. The prior dataset audit found no full-row or full-predictor duplicates, but strict five-feature profiles repeat.
- **Verified correct:** no SMOTE or other resampling is used; threshold 0.5 is prespecified.

Fold audit summary:

 repeat  fold  seed  train_rows  test_rows  train_test_overlap  stratified_train_cad  stratified_test_cad   status
      1     1    42         242         61                   0                   172                   44 verified
      1     2    42         242         61                   0                   173                   43 verified
      1     3    42         242         61                   0                   173                   43 verified
      1     4    42         243         60                   0                   173                   43 verified
      1     5    42         243         60                   0                   173                   43 verified
      2     1    43         242         61                   0                   172                   44 verified
      2     2    43         242         61                   0                   173                   43 verified
      2     3    43         242         61                   0                   173                   43 verified
      2     4    43         243         60                   0                   173                   43 verified
      2     5    43         243         60                   0                   173                   43 verified
      3     1    44         242         61                   0                   172                   44 verified
      3     2    44         242         61                   0                   173                   43 verified
      3     3    44         242         61                   0                   173                   43 verified
      3     4    44         243         60                   0                   173                   43 verified
      3     5    44         243         60                   0                   173                   43 verified
      4     1    45         242         61                   0                   172                   44 verified
      4     2    45         242         61                   0                   173                   43 verified
      4     3    45         242         61                   0                   173                   43 verified
      4     4    45         243         60                   0                   173                   43 verified
      4     5    45         243         60                   0                   173                   43 verified
      5     1    46         242         61                   0                   172                   44 verified
      5     2    46         242         61                   0                   173                   43 verified
      5     3    46         242         61                   0                   173                   43 verified
      5     4    46         243         60                   0                   173                   43 verified
      5     5    46         243         60                   0                   173                   43 verified

Summary: `{"repeat_count": 5, "folds_per_repeat": {"1": 5, "2": 5, "3": 5, "4": 5, "5": 5}, "outer_train_test_overlaps": 0, "test_coverage_violations": 0, "prediction_rows": 13635, "inner_assignments_saved": false, "inner_assignments_reconstructible_from_source_seed_formula": true, "patient_grouping": "not independently verifiable: no patient identifier is present in the Z-Alizadeh derivative"}`.
