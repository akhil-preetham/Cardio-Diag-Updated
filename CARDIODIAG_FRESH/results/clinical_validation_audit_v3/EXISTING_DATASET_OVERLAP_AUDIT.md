# Existing Dataset Overlap Audit v3

Exact normalized common-feature fingerprints were calculated where schemas permitted. A matching clinical profile is not treated as patient identity. Cleveland and Statlog show confirmed record-level correspondence; other source relationships remain unresolved without source-level provenance or identifiers. Dataset names and different row counts do not establish independence.

{
  "pairwise_checks": [
    {
      "left": "cleveland",
      "right": "statlog",
      "common_predictor_fields": 13,
      "exact_common_predictor_matches": 270,
      "classification": "confirmed_record_level_overlap",
      "matched_right_rows": 270,
      "target_agreement_after_standard_mapping": 270,
      "evidence": "All 270 Statlog predictor profiles exactly match unique Cleveland predictor profiles; mapped targets agree for all 270 rows."
    },
    {
      "left": "cleveland",
      "right": "five_states",
      "common_predictor_fields": 11,
      "exact_common_predictor_matches": 0,
      "classification": "no_exact_match_detected_under_normalization"
    },
    {
      "left": "cleveland",
      "right": "mendeley",
      "common_predictor_fields": 12,
      "exact_common_predictor_matches": 0,
      "classification": "no_exact_match_detected_under_normalization"
    },
    {
      "left": "cleveland",
      "right": "kaggle_cvd_70000",
      "common_predictor_fields": 2,
      "exact_common_predictor_matches": 0,
      "classification": "unresolvable_schema_or_provenance"
    },
    {
      "left": "statlog",
      "right": "mendeley",
      "common_predictor_fields": 12,
      "exact_common_predictor_matches": 0,
      "classification": "no_exact_match_detected_under_normalization"
    }
  ],
  "interpretation": "Identical clinical profiles are not treated as patient identity; no independent-validation claim is permitted."
}