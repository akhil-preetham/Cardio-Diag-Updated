# Executive Summary

This audit independently verified local provenance, hashes, official checksum matches, ZIP integrity, schemas, target definitions, feature eligibility, NHANES merge/design variables, ECG patient-level structure, and cross-dataset overlap limitations. No model training, tuning, or evaluation was performed.

The scientifically defensible primary research track is a leakage-controlled Z-Alizadeh Sani CAD analysis using only pre-diagnostic predictors. NHANES is suitable for weighted descriptive/associational analyses, not prospective prediction. PTB-XL and PTB Diagnostic are separate ECG tracks with patient-level splitting and unresolved overlap. Auricle remains metadata-only.

NHANES merged participants: 9254; duplicate SEQN rows: 0. Z-Alizadeh excluded direct/downstream fields are recorded in Z_ALIZADEH_FEATURE_AUDIT.csv.
