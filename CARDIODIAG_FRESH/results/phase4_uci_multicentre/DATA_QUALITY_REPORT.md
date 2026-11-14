# Data quality report

The cohort contains 920 rows and 509 positive outcomes. Exact duplicate analysis found 2 duplicate occurrences in 2 groups. Two groups are within-site repeats; no patient identifier exists, so record-level deduplication cannot establish patient-level independence. Missingness is centre-dependent and is handled inside training partitions only. Zero values are retained because the public source does not provide sufficient field-level provenance to convert them automatically to missing.
