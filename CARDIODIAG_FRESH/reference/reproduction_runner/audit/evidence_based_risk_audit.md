# Evidence-based second-level risk audit

Generated: `2026-11-12T18:17:52+00:00`

This is a source-only audit. Notebook kernels were not started and no notebook cells were executed.
Statuses mean **confirmed** (the cited source establishes the behavior), **suspected** (the source indicates a risk but the full runtime path or invariant is not established), or **not established** (the static evidence is insufficient).

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 9: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Apply oversampling using SMOTE to balance the dataset
smote = SMOTE()
```
- What the code does: SMOTE is fitted before the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 5: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
df['ChestPainType'] = chest_pain_encoder.fit_transform(df['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 15: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 19: Test-set influence
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Fit the model with early stopping
n_epochs = 100
```
- What the code does: The source contains an explicit test-set selection/stopping pattern.
- Why potentially problematic: Using test performance to choose a model, threshold, or stopping point invalidates the final test estimate.
- Exact remediation: Select models and thresholds using training CV only, then evaluate the test set once.

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 5: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 20: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb — cell 2: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 43: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# FULL PIPELINE: preprocess → SMOTE → voting classifier
clf_pipeline = Pipeline(steps=[
```
- What the code does: The first visible SMOTE operation follows the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 26: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 26: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 3: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 14: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 29: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
y_true_col = "HeartDisease"
threshold = 0.5  # classification threshold
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Fairness_OSEC_FiveStates_dataset.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 46: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
)
# FULL PIPELINE: preprocess → SMOTE → voting classifier
clf_pipeline = Pipeline(steps=[
```
- What the code does: The first visible SMOTE operation follows the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 27: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 27: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 3: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 14: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 27: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
voting_probabilities = voting_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities)
auc_score = auc(fpr, tpr)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Fairness_OVEC_FiveStates_dataset.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 1: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 21: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 21: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 13: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 66: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
voting_probabilities = voting_model.predict_proba(X_test_five)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_five, voting_probabilities)
auc_score = auc(fpr, tpr)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 1: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 25: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 25: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 17: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 56: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
ys42_probabilities = calibrated_stacking_rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys42, tpr_ys42, thresholds = roc_curve(y_test, ys42_probabilities)
    auc_score = auc(fpr_ys42, tpr_ys42)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 2: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 8: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 34: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 34: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 6: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 24: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 38: LIME held-out observations
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Create a LIME explainer with actual feature names
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,  # Training data
```
- What the code does: LIME explanation code is present; static evidence does not prove whether explained rows were held out or used for model selection.
- Why potentially problematic: Explaining training or selected observations can overstate explanation generality and may reuse evaluation data.
- Exact remediation: Record the exact explanation cohort, require held-out observations, and keep explanation generation after model selection.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 2: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 8: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 32: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
voting_probabilities = voting_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities)
    auc_score = auc(fpr, tpr)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb — cell 2: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 2: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 8: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 31: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 31: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 6: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 30: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
def NXAIH_DRC_modular(
        X, y, model,
```
- What the code does: XAIH-DRC is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 25: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 30: LIME held-out observations
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
if use_LIME:
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
```
- What the code does: LIME explanation code is present; static evidence does not prove whether explained rows were held out or used for model selection.
- Why potentially problematic: Explaining training or selected observations can overstate explanation generality and may reuse evaluation data.
- Exact remediation: Record the exact explanation cohort, require held-out observations, and keep explanation generation after model selection.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 2: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 8: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 64: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
def overlap_by_threshold(global_importance, heart_disease_corr, T_importance, T_corr):
    """
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\OVEC_Reviewer1_Five_States_Full_Methodology.ipynb — cell 2: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 2: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 8: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 43: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 43: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 6: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 24: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 48: LIME held-out observations
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Create a LIME explainer with actual feature names
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,  # Training data
```
- What the code does: LIME explanation code is present; static evidence does not prove whether explained rows were held out or used for model selection.
- Why potentially problematic: Explaining training or selected observations can overstate explanation generality and may reuse evaluation data.
- Exact remediation: Record the exact explanation cohort, require held-out observations, and keep explanation generation after model selection.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 2: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 8: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 41: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
voting_probabilities = voting_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities)
    auc_score = auc(fpr, tpr)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Reviewer1_Comment_15_LIME__All.ipynb — cell 2: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 2: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 8: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 35: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 35: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 6: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 52: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# -------------------------------
# 2. Apply NXAIH-DRC Selection
# -------------------------------
```
- What the code does: XAIH-DRC is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 25: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 35: LIME held-out observations
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
```
- What the code does: LIME explanation code is present; static evidence does not prove whether explained rows were held out or used for model selection.
- Why potentially problematic: Explaining training or selected observations can overstate explanation generality and may reuse evaluation data.
- Exact remediation: Record the exact explanation cohort, require held-out observations, and keep explanation generation after model selection.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 2: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 8: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 47: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
def overlap_by_threshold(global_importance, heart_disease_corr, T_importance, T_corr):
    """
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb — cell 2: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 2: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 8: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 35: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 35: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 6: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 52: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# -------------------------------
# 2. Apply NXAIH-DRC Selection
# -------------------------------
```
- What the code does: XAIH-DRC is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 25: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 35: LIME held-out observations
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
```
- What the code does: LIME explanation code is present; static evidence does not prove whether explained rows were held out or used for model selection.
- Why potentially problematic: Explaining training or selected observations can overstate explanation generality and may reuse evaluation data.
- Exact remediation: Record the exact explanation cohort, require held-out observations, and keep explanation generation after model selection.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 2: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 8: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 47: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
def overlap_by_threshold(global_importance, heart_disease_corr, T_importance, T_corr):
    """
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Reviewer1_XAIH_DRC.ipynb — cell 2: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 35: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
smote = SMOTE(
    sampling_strategy='auto',  # balances all minority classes
```
- What the code does: The first visible SMOTE operation follows the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 19: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 19: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 14: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 11: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Function to mark outliers as NaN using Z-score
def mark_outliers_zscore(df, numerical_cols, z_threshold):
    df_copy = df.copy()
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Reviewer2_NOSEC_5state_dataset_Concern_22_and_18+Ablation_Results.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 1: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 24: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 24: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 95: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data_old[data_old.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 121: OVEC/OSEC implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated Proposed OVEC')
plt.plot([0, 1], [0, 1], '--', color='black', label='Perfect Calibration')
```
- What the code does: OVEC is referenced in the notebook source; static inspection does not prove that the claimed algorithm is implemented as described or executed end-to-end.
- Why potentially problematic: A label, filename, or result table can be mistaken for an implementation, and leakage or post-test tuning may remain hidden in helper code.
- Exact remediation: Map the paper definition to executable functions and verify each input/output; implement an independent corrected comparator rather than editing the original.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 85: OVEC/OSEC implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', label="Platt Calibrated Proposed OSEC")
#plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
```
- What the code does: OSEC is referenced in the notebook source; static inspection does not prove that the claimed algorithm is implemented as described or executed end-to-end.
- Why potentially problematic: A label, filename, or result table can be mistaken for an implementation, and leakage or post-test tuning may remain hidden in helper code.
- Exact remediation: Map the paper definition to executable functions and verify each input/output; implement an independent corrected comparator rather than editing the original.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 17: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 26: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
ys51_probabilities = best_tree.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys51, tpr_ys51, thresholds = roc_curve(y_test, ys51_probabilities)
    auc_score = auc(fpr_ys51, tpr_ys51)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 7: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Apply oversampling using SMOTE to balance the dataset
smote = SMOTE()
```
- What the code does: SMOTE is fitted before the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 10: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree_daza = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree_daza.fit(X_train_daza, y_train_daza)
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 10: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree_daza = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree_daza.fit(X_train_daza, y_train_daza)
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 54: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data_old[data_old.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 76: OVEC/OSEC implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
plt.plot(fpr_novec, tpr_novec, linestyle='--', marker='P', markevery=15,
         label=f"Proposed OVEC (Optimized Voting Ensemble Classifier) (AUC = {auc_novec:.4f})")
```
- What the code does: OVEC is referenced in the notebook source; static inspection does not prove that the claimed algorithm is implemented as described or executed end-to-end.
- Why potentially problematic: A label, filename, or result table can be mistaken for an implementation, and leakage or post-test tuning may remain hidden in helper code.
- Exact remediation: Map the paper definition to executable functions and verify each input/output; implement an independent corrected comparator rather than editing the original.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 76: OVEC/OSEC implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
plt.plot(fpr_nosec, tpr_nosec, linestyle='-', marker='*', markevery=15,
         label=f"Proposed OSEC (Optimized Stacking Ensemble Classifier) (AUC = {auc_nosec:.4f})")
```
- What the code does: OSEC is referenced in the notebook source; static inspection does not prove that the claimed algorithm is implemented as described or executed end-to-end.
- Why potentially problematic: A label, filename, or result table can be mistaken for an implementation, and leakage or post-test tuning may remain hidden in helper code.
- Exact remediation: Map the paper definition to executable functions and verify each input/output; implement an independent corrected comparator rather than editing the original.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 22: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 14: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Plot ROC Curve
fpr_daza, tpr_daza, thresholds_daza = roc_curve(y_test_daza, y_pred_proba_daza)
plt.figure()
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 2: SMOTE split order
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```
- What the code does: SMOTE is mentioned, but no active fit_resample call and split order were both found statically.
- Why potentially problematic: A name or import does not prove that resampling was executed.
- Exact remediation: Trace the active call path before reproduction; use a training-fold pipeline in corrected validation.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 8: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 43: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 43: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 6: Identifier or duplicate leakage
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Check for duplicates
duplicates = data[data.duplicated()]
```
- What the code does: Duplicate checking/removal is referenced, but patient-level independence is not established.
- Why potentially problematic: Row-level duplicate checks do not prove that the same patient cannot occur in both partitions.
- Exact remediation: Verify patient identifiers and apply grouped or deduplicated splitting before corrected validation.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 24: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 48: LIME held-out observations
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Create a LIME explainer with actual feature names
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,  # Training data
```
- What the code does: LIME explanation code is present; static evidence does not prove whether explained rows were held out or used for model selection.
- Why potentially problematic: Explaining training or selected observations can overstate explanation generality and may reuse evaluation data.
- Exact remediation: Record the exact explanation cohort, require held-out observations, and keep explanation generation after model selection.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 2: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 8: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 41: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
voting_probabilities = voting_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities)
    auc_score = auc(fpr, tpr)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb — cell 2: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 50: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# FULL PIPELINE: preprocess → SMOTE → voting classifier
clf_pipeline = Pipeline(steps=[
```
- What the code does: The first visible SMOTE operation follows the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 15: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
df_heart_disease[numerical_cols] = median_imputer.fit_transform(df_heart_disease[numerical_cols])
df_normal[numerical_cols] = median_imputer.fit_transform(df_normal[numerical_cols])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 33: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 33: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 6: Identifier or duplicate leakage
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Drop Unwanted Columns
mendeley.drop(columns=["patientid", "noofmajorvessels"], inplace=True, errors="ignore")
mendeley.shape
```
- What the code does: An identifier-like field or identifier handling is present in the source.
- Why potentially problematic: Identifiers can memorize subjects or permit patient overlap across partitions; duplicate handling may also alter the cohort.
- Exact remediation: Audit identifier uniqueness and duplicates, exclude identifiers from predictors, and use grouped splitting when repeated patients exist.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 18: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 5: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
rename_dict = {
    "age": "Age",
    "gender": "Sex",
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 36: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
y_true_col = "HeartDisease"
threshold = 0.5  # classification threshold
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 52: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
)
# FULL PIPELINE: preprocess → SMOTE → voting classifier
clf_pipeline = Pipeline(steps=[
```
- What the code does: The first visible SMOTE operation follows the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 15: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
df_heart_disease[numerical_cols] = median_imputer.fit_transform(df_heart_disease[numerical_cols])
df_normal[numerical_cols] = median_imputer.fit_transform(df_normal[numerical_cols])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 33: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 33: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 6: Identifier or duplicate leakage
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Drop Unwanted Columns
mendeley.drop(columns=["patientid", "noofmajorvessels"], inplace=True, errors="ignore")
mendeley.shape
```
- What the code does: An identifier-like field or identifier handling is present in the source.
- Why potentially problematic: Identifiers can memorize subjects or permit patient overlap across partitions; duplicate handling may also alter the cohort.
- Exact remediation: Audit identifier uniqueness and duplicates, exclude identifiers from predictors, and use grouped splitting when repeated patients exist.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 18: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 5: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
rename_dict = {
    "age": "Age",
    "gender": "Sex",
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 33: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
voting_probabilities = voting_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities)
auc_score = auc(fpr, tpr)
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## mendeley_results\MENDELEY Results\Fairness_OVEC_MENDELEY_dataset.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 29: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Apply oversampling using SMOTE to balance the dataset
smote = SMOTE()
```
- What the code does: SMOTE is fitted before the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 19: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
df_heart_disease[numerical_cols] = median_imputer.fit_transform(df_heart_disease[numerical_cols])
df_normal[numerical_cols] = median_imputer.fit_transform(df_normal[numerical_cols])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 32: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree_daza = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree_daza.fit(X_train_daza, y_train_daza)
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 32: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
tree = DecisionTreeClassifier()
grid_search_tree_daza = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree_daza.fit(X_train_daza, y_train_daza)
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 7: Identifier or duplicate leakage
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Drop Unwanted Columns
mendeley.drop(columns=["patientid", "noofmajorvessels"], inplace=True, errors="ignore")
mendeley.shape
```
- What the code does: An identifier-like field or identifier handling is present in the source.
- Why potentially problematic: Identifiers can memorize subjects or permit patient overlap across partitions; duplicate handling may also alter the cohort.
- Exact remediation: Audit identifier uniqueness and duplicates, exclude identifiers from predictors, and use grouped splitting when repeated patients exist.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 136: OVEC/OSEC implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
print(f"Calibration Intercept (OVEC): {calib_intercept:.4f}")
print(f"Calibration Slope (OVEC):     {calib_slope:.4f}")
```
- What the code does: OVEC is referenced in the notebook source; static inspection does not prove that the claimed algorithm is implemented as described or executed end-to-end.
- Why potentially problematic: A label, filename, or result table can be mistaken for an implementation, and leakage or post-test tuning may remain hidden in helper code.
- Exact remediation: Map the paper definition to executable functions and verify each input/output; implement an independent corrected comparator rather than editing the original.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 77: OVEC/OSEC implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
print("========== Platt-Calibrated OSEC Model ==========")
print(f"Training Accuracy       : {train_accuracy:.4f}")
```
- What the code does: OSEC is referenced in the notebook source; static inspection does not prove that the claimed algorithm is implemented as described or executed end-to-end.
- Why potentially problematic: A label, filename, or result table can be mistaken for an implementation, and leakage or post-test tuning may remain hidden in helper code.
- Exact remediation: Map the paper definition to executable functions and verify each input/output; implement an independent corrected comparator rather than editing the original.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 22: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 6: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
rename_dict = {
    "age": "Age",
    "gender": "Sex",
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 36: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Plot ROC Curve
fpr_daza, tpr_daza, thresholds_daza = roc_curve(y_test_daza, y_pred_proba_daza)
plt.figure()
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 35: SMOTE split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
smote = SMOTE(
    sampling_strategy='auto',  # balances all minority classes
```
- What the code does: The first visible SMOTE operation follows the first train/test split.
- Why potentially problematic: Resampling before splitting allows synthetic information derived from eventual test observations to enter training.
- Exact remediation: For corrected validation, split first and put SMOTE inside each training fold. Preserve the original ordering in the author-faithful copy.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 4: Preprocessing split order
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])
```
- What the code does: A transformer/model-like fit operation appears before the first train/test split.
- Why potentially problematic: Fitting encoders, scalers, imputers, PCA, or feature selection on all rows can leak test-distribution information.
- Exact remediation: Fit preprocessing only on training folds through a pipeline; do not rewrite the author notebook.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 19: Cross-validation nesting
- Status: **confirmed**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: An inner CV search object is present alongside an outer train/test split; the visible search is not nested inside an outer evaluation loop.
- Why potentially problematic: Non-nested selection can produce optimistic estimates if the same evaluation data are reused for model selection or reporting.
- Exact remediation: Use nested CV or reserve an untouched test set and perform all search only on training data.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 19: Test-set influence
- Status: **not established**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
```
- What the code does: Model-selection code is present, but direct test-set selection was not established by this static rule.
- Why potentially problematic: Saved outputs or later cells may still have influenced choices.
- Exact remediation: Review the complete execution path and keep test data out of all selection decisions.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 14: XAIH-DRC/LoRAS implementation
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras
```
- What the code does: LoRAS is actively referenced in source.
- Why potentially problematic: The name alone does not establish a faithful algorithm, parameterization, or execution path.
- Exact remediation: Verify the implementation against the paper and record package/version/input/output details; do not modify the source notebook.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 1: Paired statistical predictions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
```
- What the code does: Bootstrap or statistical comparison code is present.
- Why potentially problematic: Paired tests require predictions from identical observations and a prespecified pairing; static names do not prove that invariant.
- Exact remediation: Persist row identifiers and prediction vectors for every model, verify identical test indices, then run paired tests only on matched observations.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 4: Fairness subgroup validity
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
```
- What the code does: Subgroup/fairness analysis code is present.
- Why potentially problematic: Subgroup metrics can be unstable or misleading without prespecified groups, minimum sample sizes, confidence intervals, and multiplicity handling.
- Exact remediation: Define clinical/demographic groups before analysis, report counts, enforce minimum sizes, calculate uncertainty, and avoid post hoc subgroup selection.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 11: Clinical utility assumptions
- Status: **suspected**
- Affects: author-faithful and corrected validation
- Source excerpt:
```python
# Function to mark outliers as NaN using Z-score
def mark_outliers_zscore(df, numerical_cols, z_threshold):
    df_copy = df.copy()
```
- What the code does: Decision-curve, threshold, or clinical-utility code is present.
- Why potentially problematic: Clinical utility depends on defensible threshold probabilities, prevalence, harms, benefits, calibration, and an appropriate target population.
- Exact remediation: Document threshold and utility assumptions, validate calibration and prevalence externally, and avoid clinical claims from internal test data alone.

## mendeley_results\MENDELEY Results\Reviewer2_NOSEC_Mendeley_dataset_Ablation_Results.ipynb — cell 1: External or destructive operations
- Status: **confirmed**
- Affects: author-faithful reproduction only
- Source excerpt:
```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```
- What the code does: The notebook mounts Colab Drive, installs packages, invokes shell/network/process operations, or deletes files.
- Why potentially problematic: Execution can depend on unavailable external files, mutate the environment, overwrite artifacts, or remove data.
- Exact remediation: Do not execute the original. Use a copied notebook, pre-approved local paths, isolated outputs, and manual review of every external/destructive operation.
