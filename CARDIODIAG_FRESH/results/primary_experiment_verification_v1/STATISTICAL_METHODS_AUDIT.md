# Statistical Methods Audit

- **Verified correct:** no ordinary independent-samples t-test or Wilcoxon test is used on correlated CV fold scores.
- **Correct with limitations:** comparisons use paired observation-level bootstrap differences after averaging each model's repeated-CV predictions by source row. This avoids treating the repeated predictions for a row as separate observations.
- **Requires correction in interpretation:** the comparison is not a full repeated-CV corrected test of algorithmic performance because it conditions on averaged predictions and does not model split or fitting variation.
- **Verified correct:** p-values are intentionally absent and no multiple-comparison-adjusted significance claim is made.
- **Unresolved:** the bootstrap comparison does not provide a population-level inferential test for a new clinical cohort.

The saved comparison file contains `8` descriptive ROC-AUC comparisons against unweighted logistic regression.
