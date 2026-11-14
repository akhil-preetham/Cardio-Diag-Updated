# CARDIODIAG dataset catalog

## five_states

- Title: Five States / Heart Disease Prediction
- Source: Unresolved local provenance
- DOI: none
- License: Unknown; do not redistribute beyond project permissions
- Rows × columns: 918 × 12
- Target: `HeartDisease`; role: original baseline disease-presence dataset

## mendeley

- Title: Cardiovascular_Disease_Dataset
- Source: https://data.mendeley.com/datasets/dzz48mvjht/1
- DOI: 10.17632/dzz48mvjht.1
- License: CC BY 4.0
- Rows × columns: 1000 × 14
- Target: `target`; role: original baseline disease-presence dataset

## kaggle_cvd_70000

- Title: Cardiovascular Disease Dataset
- Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- DOI: none
- License: Kaggle dataset license not independently resolved; retain attribution and check terms before redistribution
- Rows × columns: 70000 × 13
- Target: `cardio`; role: additional disease-presence classification benchmark

## uci_heart_disease_cleveland

- Title: Heart Disease
- Source: https://archive.ics.uci.edu/dataset/45/heart+disease
- DOI: 10.24432/C52P4X
- License: CC BY 4.0
- Rows × columns: 303 × 14
- Target: `num`; role: additional/source-held-out disease-presence benchmark

## statlog_heart

- Title: Statlog (Heart)
- Source: https://archive.ics.uci.edu/dataset/145/statlog+heart
- DOI: 10.24432/C57303
- License: CC BY 4.0
- Rows × columns: 270 × 14
- Target: `target`; role: additional disease-presence benchmark; overlap with UCI Heart must be assessed

## uci_heart_failure

- Title: Heart Failure Clinical Records
- Source: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
- DOI: 10.24432/C5Z89R
- License: CC BY 4.0
- Rows × columns: 299 × 13
- Target: `DEATH_EVENT`; role: separate cardiovascular outcome task; not disease-presence classification

Raw files are preserved under `data/raw/` or their existing reference path. See the registry and hashes for exact provenance.
