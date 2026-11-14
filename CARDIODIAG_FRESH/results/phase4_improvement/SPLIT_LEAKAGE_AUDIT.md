# Split and leakage audit

Each saved prediction is keyed by feature set, model, held-out centre and source row. The held-out centre is absent from the development data by construction. The saved Phase 4 implementation fits imputation, scaling and hyperparameter search inside development-centre data. No held-out labels are used for feature selection, imputation, scaling, calibration or threshold selection. Record-level bootstrap intervals remain limited because patient identifiers and centre-cluster resampling are unavailable.
