# Model selection protocol

The primary reference remains regularized logistic regression with fold-local preprocessing and a bounded C/class-weight grid. Random forest and histogram gradient boosting results are inherited from the verified Phase 4 comparison. Firth, GAM and Bayesian hierarchical models were not executed: no validated implementation was already present, and installing unverified modelling dependencies would not provide defensible evidence on this small, highly shifted cohort. This is an explicit feasibility limitation, not a claim that those methods cannot be useful.
