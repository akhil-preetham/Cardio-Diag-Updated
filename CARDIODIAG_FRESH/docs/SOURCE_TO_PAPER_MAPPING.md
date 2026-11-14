# Source-to-paper mapping

## Publication metadata

Crossref metadata for DOI `10.7717/peerj-cs.4059` identifies:

> Khan, Muhammad Jaffar; Raza, Basit; Nadeem, Adnan (2026). “CardioDiag: a machine learning approach for cardiovascular disease diagnosis using XAI-guided hybrid dimensionality reduction and optimized-based ensemble classifiers.” *PeerJ Computer Science*, volume 12, article e4059. DOI: `10.7717/peerj-cs.4059`.

The local PDF is `C:\Users\akhil\Downloads\Final_Year_Project\peerj-cs-4059-BASE_PAPER.pdf`. The requested alternate filename with `(5)` is absent. No local PDF text-extraction executable or installed extraction library was available, so exact paper tables, figures, captions, and full reference text could not be independently transcribed. The abstract-level claims below are from Crossref metadata; they are not treated as exact table values.

## Claim mapping

| Paper claim or method | Source evidence | Our evidence | Comparability |
|---|---|---|---|
| Percentile outlier/skewness handling | Static notebook audit; exact implementation path not fully verified | Not part of corrected runner | Not comparable |
| LoRAS class balancing | `loras` references in supplementary notebooks | No end-to-end execution; dependency/artifact path unresolved | Not reproduced |
| XAIH-DRC combining correlation and LIME importance | `Reviewer1_XAIH_DRC.ipynb` and related notebooks | Static source evidence only | Not reproduced |
| OVEC and OSEC ensemble classifiers | OVEC/OSEC notebook family | Corrected runner uses a separate stacking comparator, not author OVEC/OSEC | Not comparable |
| Five States base accuracy | Executed prepared base notebook | Accuracy 0.8922; SMOTE-before-split and test-driven stopping preserved | Author-faithful local run, not proven historical equivalence |
| Five States corrected comparator | `five_states_corrected_validation_report.json` | Accuracy 0.8783 mean; ROC-AUC 0.9332 mean | Different split, preprocessing, SMOTE, selection, and repeated design |
| Mendeley corrected comparator | `mendeley_corrected_validation_report.json` | Accuracy 0.9772 mean; ROC-AUC 0.9964 mean; `patientid` excluded | Different protocol; not a paper reproduction |
| Calibration and decision-curve claims | Static notebook evidence and paper abstract | Brier/ECE summaries only for corrected predictions; no validated decision-curve utility | Not comparable |

## Published-versus-measured comparison

The available abstract states approximate relative gains for OSEC but does not provide exact dataset/model/table values in the metadata response. Exact published values are therefore recorded as **not verified**, rather than inferred.

| Dataset/model/metric | Published value | Our measured value | Protocol | Difference |
|---|---|---|---|---|
| Five States / author-faithful base / accuracy | Exact table value not verified | 0.8922 | SMOTE before 80:20 split; one run; test-driven stopping | Not calculable |
| Five States / corrected comparator / accuracy | Exact table value not verified | 0.8783 mean | Repeated stratified 75:25; fold-local preprocessing and SMOTE; 10 overlapping repeats | Not calculable and not directly comparable |
| Five States / corrected comparator / ROC-AUC | Exact table value not verified | 0.9332 mean | Same corrected protocol | Not calculable and not directly comparable |
| Mendeley / corrected comparator / accuracy | Exact table value not verified | 0.9772 mean | `patientid` excluded; corrected protocol | Not calculable and not directly comparable |

No paper-reported metric is labelled independently reproduced unless it came from an executed, hashed project run.
