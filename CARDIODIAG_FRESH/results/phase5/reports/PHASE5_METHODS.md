# Phase 5 methods

All preprocessing, imputation, encoding, scaling, spline transformation and model tuning occur inside development folds. ZAS uses repeated stratified nested validation. UCI uses leave-one-centre-out outer validation with development-only inner selection. Saved Phase 4.3 predictions are used for independent staged reliability comparisons. Record-level bootstrap intervals do not represent patient-cluster or centre-sampling uncertainty.
