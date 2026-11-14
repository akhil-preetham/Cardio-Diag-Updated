# Dataset and provenance audit

DATASET_DIMENSIONS.csv contains newly counted local CSV dimensions and exact serialized-row duplicates for validated datasets. Missingness normalization, near duplicates, cross-dataset overlap, licenses and patient identity are not independently resolved by these counts.

| Track | Evidence and restrictions |
|---|---|
| ZAS | Local validated CSV; Phase 6.1 tests reconcile source labels, partitions and predictions. Cath Cad=1/Normal=0; 303 source rows, 216 positive, 87 negative. Local 56 columns including target imply 55 other columns; source catalogue's 54-feature description remains unresolved. No verified independent-patient identifiers or feature acquisition times |
| Mendeley | Separate validated/raw CSVs and metadata; patientid excluded in corrected historical workflow according to README. Historical 1,000-row claim must be paired with current counted dimensions and original provenance |
| Five States/Heart5 | Separate historical reproduction track; parent heart5.csv and validated five_states.csv. File equivalence is not assumed solely from similar dimensions |
| Kaggle 70k | Separate cardiovascular endpoint and computationally reduced historical workflow; no transfer of its scores into Phase 6 |
| UCI four-centre | 920 records documented in manuscript/audits; 509 positive/411 negative, num 0 versus 1–4. Four-centre source reconstruction is distinct from validated Cleveland-only CSV |
| Other resources | Statlog, heart failure and ECG/clinical metadata exist; distinct outcomes and modalities. Their presence does not make them compatible external cohorts |

Original paper/repository/Mendeley/Zenodo relationships and bibliography were not freshly checked against authoritative online sources in this initial local pass. They remain unresolved rather than accepted as final citations. Technical local verification is separate from publication permission or ethics determinations.
