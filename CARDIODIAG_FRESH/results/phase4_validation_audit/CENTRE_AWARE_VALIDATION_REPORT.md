# Centre-aware inner validation

The historical staged implementation selects logistic hyperparameters using four shuffled stratified folds within the development centres. This audit repeats each outer holdout using either that procedure or leave-one-development-centre-out inner validation. The outer centre remains untouched in both procedures. The comparison is diagnostic, not a model-selection rule: outer performance is reported only after the method is fixed.
