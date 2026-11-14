# Model Updating Protocol

First report untouched V3 external validation. Only afterward consider intercept recalibration, logistic recalibration of intercept and slope, or coefficient updating. Freeze the updating question and parameter count before fitting. Use internal resampling within the external cohort to quantify optimism, and reserve a final validation portion if substantial updating occurs. Never use external performance to retroactively select the original model. Any updated model is a new model with a new identifier and separate validation evidence.
