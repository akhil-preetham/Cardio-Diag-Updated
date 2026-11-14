# Phase H — Interpretability evidence

For the frozen V3 reference, inspect fold-level logistic coefficient/permutation summaries in `results/primary_research_v3/FEATURE_IMPORTANCE_RESULTS.csv`. For the UCI research track, source reports provide logistic model results and feature effects; the final ZAS exploratory output was not selected as a deployable model. These importances describe model associations under the fitted pipeline and observed cohort. They do not establish causation or stable clinical effect.

No local explanation is used to choose a final feature set or tune against held-out outcomes. The legacy CardioDiag LIME outputs concern separate Five States/Mendeley experiments and are not transferred as explanations for ZAS or UCI models. Fold-level stability, coding, collinearity and timing limits must accompany any coefficient or permutation-importance display. No local explanation is claimed for an unavailable externally validated model.
