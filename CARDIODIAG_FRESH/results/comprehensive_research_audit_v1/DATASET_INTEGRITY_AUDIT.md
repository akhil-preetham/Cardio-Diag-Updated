# Dataset integrity audit

## Inventory

| Dataset | Rows | Columns | Target | Class distribution | Missing values |
|---|---:|---:|---|---|---|
| `reference/author_release/heart5.csv` | 918 | 12 | `HeartDisease` | 0:410, 1:508 | 0 in every column |
| `reference/author_release/MendeleyCardiovascular_Disease_Dataset.csv` | 1,000 | 14 | `target` | 0:420, 1:580 | 0 in every column |
| Approximately 70,000-row benchmark | unavailable | unavailable | unavailable | unavailable | not assessed |

Hashes and complete dtype/column metadata are in the prior duplicate-audit artifacts and this audit's manifest.

## Duplicates and identifiers

The duplicate audit found zero exact duplicate groups, zero predictor-only duplicate groups, zero conservative near-duplicate candidates, and zero repeated Mendeley `patientid` values. No duplicate or identical-predictor overlap occurred across the ten reproduced train/test holdouts for either dataset. No observations were removed.

This evidence does not establish patient identity from clinical attributes and does not prove that no patient overlap exists in an unavailable external source.

## Validity and distribution

The supplied CSVs have complete values and the expected row counts. The existing dataset validation and LoRAS reports should be used for detailed range/outlier evidence. The current audit does not claim that a plausible numeric range proves clinical correctness or absence of collection bias.

## Mendeley performance

Corrected validation achieved mean ROC-AUC 0.9964 (SD 0.0028), with `patientid` excluded. Duplicate leakage is not supported by the available row-level and split-overlap evidence. The unusually high result therefore remains unexplained by duplicates alone; dataset separability, target construction, feature distributions, and source/population characteristics require separate investigation.
