# Explainability and Stability Audit

## Classification

- **Verified correct:** permutation importance was computed from fitted pipelines on the corresponding outer test partitions (`primary_research_v1.py`, `run_experiment`).
- **Correct with limitations:** the held-out labels are required to score permutation importance, so this is a post-hoc explanatory analysis, not a model-selection input.
- **Correct with limitations:** importance is predictive association, not causal effect. Correlated and derived clinical variables can redistribute importance; the strict set is small but not immune to this issue.
- **Unresolved:** no patient identifier is available to assess subject-level grouping beyond the dataset documentation and prior integrity audit.

## Stability

Importance rows: 1125; expected five features across nine configurations and 25 outer folds. The following top-three summaries are descriptive and should not be interpreted as causal rankings:

                    model_id        feature  mean_importance  sd_importance  positive_fraction  folds
        extra_trees_balanced            Age         0.192302       0.068633               1.00     25
        extra_trees_balanced Current Smoker         0.032697       0.042654               0.80     25
        extra_trees_balanced            Sex         0.020950       0.034224               0.72     25
            extra_trees_none            Age         0.195406       0.067688               1.00     25
            extra_trees_none Current Smoker         0.027833       0.040739               0.68     25
            extra_trees_none            Sex         0.014709       0.043520               0.76     25
 hist_gradient_boosting_none            Age         0.185544       0.085741               1.00     25
 hist_gradient_boosting_none             FH         0.013739       0.022282               0.80     25
 hist_gradient_boosting_none Current Smoker         0.007718       0.023376               0.64     25
logistic_regression_balanced            Age         0.233267       0.071448               1.00     25
logistic_regression_balanced             FH         0.015101       0.016321               0.76     25
logistic_regression_balanced Current Smoker         0.013606       0.017227               0.80     25
    logistic_regression_none            Age         0.231854       0.071814               1.00     25
    logistic_regression_none Current Smoker         0.013520       0.016865               0.84     25
    logistic_regression_none             FH         0.013392       0.015373               0.76     25
      random_forest_balanced            Age         0.204452       0.079590               1.00     25
      random_forest_balanced Current Smoker         0.009777       0.022400               0.64     25
      random_forest_balanced             FH         0.008575       0.019650               0.72     25
          random_forest_none            Age         0.207425       0.083750               1.00     25
          random_forest_none Current Smoker         0.010603       0.020931               0.68     25
          random_forest_none             FH         0.005688       0.018891               0.64     25
                svm_balanced            Age         0.216458       0.070791               1.00     25
                svm_balanced Current Smoker         0.017701       0.026004               0.80     25
                svm_balanced             FH         0.005015       0.025134               0.64     25
                    svm_none            Age         0.216071       0.074219               1.00     25
                    svm_none Current Smoker         0.039985       0.041163               0.80     25
                    svm_none             FH         0.025050       0.036729               0.76     25
