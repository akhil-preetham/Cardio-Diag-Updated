# Patient overlap audit

UCI site membership is explicit from the four processed source filenames. Exact full-row duplicate analysis found two duplicate groups, both within a single source site (one in Hungary and one in VA Long Beach), with identical targets. No exact duplicate crossed sites. This does not prove that no patient appears across centres because the public files contain no trustworthy patient identifier.

Statlog and Cleveland must remain non-independent: the existing CARDIODIAG audit established exact normalized predictor-profile overlap for all 270 Statlog records with Cleveland. The Statlog file is therefore excluded from independent external-validation claims.

AngioCAD has an explicit local patient ID in its spreadsheets. The labels table contains 413 unique IDs and the clinical table contains 377 unique IDs; no duplicate IDs were found within either table. Image/video files must be split by patient ID, never by frame, vessel or sequence.
