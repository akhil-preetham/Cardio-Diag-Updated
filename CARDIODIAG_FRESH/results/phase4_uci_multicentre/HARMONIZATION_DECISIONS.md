# Harmonization decisions

The strict feature set is age, sex, resting blood pressure, cholesterol, fasting blood sugar and resting ECG. These are represented in every processed centre file and are plausibly available at presentation, although source timing is not fully documented. Exercise-test variables are retained only in an exploratory expanded set. `ca`, `thal` and `slope` are excluded because they are diagnostic-context, post-diagnostic or insufficiently documented. No unavailable smoking or family-history fields are imputed into the strict set.
