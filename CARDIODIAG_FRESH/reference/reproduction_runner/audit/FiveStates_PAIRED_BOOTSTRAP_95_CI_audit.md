# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\five_states_results\Five States Results\FiveStates_PAIRED_BOOTSTRAP_95_CI.ipynb`
- SHA-256: `6222b1d27f23487d8fd351cef4a82069999f8c661cf81babe12efcf0375e6569`
- Cells: 66
- Saved outputs: 26

## Imports
- `google.colab`
- `imblearn.over_sampling`
- `loras`
- `matplotlib.pyplot`
- `numpy`
- `pandas`
- `scipy.stats`
- `sklearn.calibration`
- `sklearn.decomposition`
- `sklearn.ensemble`
- `sklearn.impute`
- `sklearn.linear_model`
- `sklearn.metrics`
- `sklearn.model_selection`
- `sklearn.naive_bayes`
- `sklearn.neighbors`
- `sklearn.preprocessing`
- `sklearn.svm`
- `sklearn.tree`
- `sklearn.utils`
- `time`
- `warnings`
- `xgboost`

## Method keywords
- CalibratedClassifierCV
- GridSearchCV
- LoRAS
- OSEC
- RandomizedSearchCV
- SMOTE
- StackingClassifier
- VotingClassifier
- bootstrap
- train_test_split
- wilcoxon

## Risk flags
- **credential_or_mount**: cells [1]
- **notebook_shell**: cells [13, 53]

## Cell sources

### Cell 1 — code (execution_count=None)
```python
import numpy as np  # Import numpy to use np.logspace
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2 — markdown (execution_count=None)
```text
# **Base paper dataset and model**
```

### Cell 3 — code (execution_count=None)
```python
data = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data
data.head()

```

### Cell 4 — code (execution_count=None)
```python
# Preprocessing
data = data.dropna()  # Drop rows with missing values

# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'

# Convert 'Sex' to 0 and 1 (e.g., Male = 1, Female = 0)
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})

# Encode 'ChestPainType' (use LabelEncoder for multi-class categorical variables)
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])

# Encode 'RestingECG'
resting_ecg_encoder = LabelEncoder()
data['RestingECG'] = resting_ecg_encoder.fit_transform(data['RestingECG'])

# Convert 'ExerciseAngina' to binary (e.g., Y = 1, N = 0)
data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y': 1, 'N': 0})
data['ST_Slope'] = data['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': 0})

#first 5 values of data
data.head()
```

### Cell 5 — markdown (execution_count=None)
```text
# **Step 1: Remove Label from Dataset
```

### Cell 6 — code (execution_count=None)
```python
X = data.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data['HeartDisease']
print(data)
print("X",X)
print("y",y)

#Target Variables Values
data.HeartDisease.value_counts()
```

### Cell 7 — markdown (execution_count=None)
```text
# **Anamoly/ Outllier Dection by Percentile Method**
```

### Cell 8 — code (execution_count=None)
```python
import numpy as np
import pandas as pd

# Assume 'data' is your DataFrame with shape (918, 12)
# For the purpose of this example, we will generate a sample DataFrame
# data = pd.read_csv('your_dataset.csv')

# Define percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Detect outliers based on percentiles for each column
outliers = pd.DataFrame()

# Iterate over each feature in the dataset
for column in data.columns:
    # Calculate the lower and upper percentile values for the current column
    lower_bound = np.percentile(data[column], lower_percentile)
    upper_bound = np.percentile(data[column], upper_percentile)

    # Flag outliers in this column
    outliers[column] = (data[column] < lower_bound) | (data[column] > upper_bound)

# Combine the outliers across all columns
# An outlier is detected if it's flagged in any feature
outliers['is_outlier'] = outliers.any(axis=1)

# Filter the rows that are flagged as outliers
data_outliers = data[outliers['is_outlier']]
data_no_outliers = data[~outliers['is_outlier']]

# Output the result
print("Number of outliers detected:", data_outliers.shape[0])
print("Data with outliers removed:", data_no_outliers.shape)

#Target Variables Values
data_outliers.HeartDisease.value_counts()

X = data_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data_no_outliers['HeartDisease']


#Target Variables Values
data_no_outliers.HeartDisease.value_counts()
X.shape
```

### Cell 9 — code (execution_count=None)
```python
import pandas as pd
import numpy as np

# Define the lower and upper percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Identify numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns

# Function to mark outliers as NaN
def mark_outliers_as_nan(df, cols, lower_perc, upper_perc):
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    for col in cols:
        lower_bound = np.percentile(df_copy[col], lower_perc)
        upper_bound = np.percentile(df_copy[col], upper_perc)
        df_copy[col] = np.where((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound), np.nan, df_copy[col])
    return df_copy  # Return the modified DataFrame

# Store the modified DataFrame
data_with_nans = mark_outliers_as_nan(data, numerical_cols, lower_percentile, upper_percentile)

# Check the result
print("Data after marking outliers as NaN:")
print(data_with_nans.isnull().sum())
```

### Cell 10 — code (execution_count=None)
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Assuming your Outliers dataset after removing outliers
# The label column is assumed to be named 'HeartDisease'

# Split data into features and labels
X_Outliers = data_with_nans.drop('HeartDisease', axis=1)  # Features
y_Outliers = data_with_nans['HeartDisease']               # Labels

# Separate into heart disease and normal groups
df_heart_disease = data_with_nans[data_with_nans['HeartDisease'] == 1].copy()
df_normal = data_with_nans[data_with_nans['HeartDisease'] == 0].copy()

# Verify data before imputation
print("Shape of Heart Disease group before numerical imputation: ", df_heart_disease.shape)
print("Shape of Normal group before numerical imputation: ", df_normal.shape)

# Impute Numerical Columns
numerical_cols = X_Outliers.select_dtypes(include=[np.number]).columns
median_imputer = SimpleImputer(strategy='median')

df_heart_disease[numerical_cols] = median_imputer.fit_transform(df_heart_disease[numerical_cols])
df_normal[numerical_cols] = median_imputer.fit_transform(df_normal[numerical_cols])

print("Shape of Heart Disease group after numerical imputation: ", df_heart_disease.shape)
print("Shape of Normal group after numerical imputation: ", df_normal.shape)

# Combine both groups back together after imputation
df_imputed = pd.concat([df_heart_disease, df_normal], ignore_index=True)

print("Final Imputed DataFrame: \n", df_imputed.head())
print("Missing values after imputation: \n", df_imputed.isnull().sum())
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 11 — code (execution_count=None)
```python
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'HeartDisease' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 12 — markdown (execution_count=None)
```text


# Step 2: Apply Oversampling Technoques/ Data Balancing Techniques
LORAS

```

### Cell 13 — code (execution_count=None)
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras

# Import necessary libraries
import loras
import numpy as np
import pandas as pd

# Majority and minority from TRAINING SET ONLY
df_imputed_X_majority = df_imputed_X[df_imputed_y == 1]   # e.g., HeartDisease = 1
df_imputed_X_minority = df_imputed_X[df_imputed_y == 0]   # e.g., HeartDisease = 0

print("Before LoRAS:")
print("Majority class:", df_imputed_X_majority.shape)
print("Minority class:", df_imputed_X_minority.shape)

```

### Cell 14 — code (execution_count=None)
```python
# 2️⃣ Apply LoRAS
df_imputed_X_minority_loras = loras.fit_resample( df_imputed_X_majority, df_imputed_X_minority)
print("After LoRAS:")
print("Minority class:", df_imputed_X_minority_loras.shape)
print("Majority class:", df_imputed_X_majority.shape)
```

### Cell 15 — code (execution_count=None)
```python
import numpy as np

# 3️⃣ Downsample If LoRAS Overshoots (Optional but Recommended)
if df_imputed_X_minority_loras.shape[0] > df_imputed_X_majority.shape[0]:
    np.random.seed(42)

    selected_indices = np.random.choice(
        df_imputed_X_minority_loras.shape[0],
        size=df_imputed_X_majority.shape[0],
        replace=False
    )

    df_imputed_X_minority_loras = df_imputed_X_minority_loras[selected_indices]

print("\nAfter LoRAS + Downsampling:")
print("Minority class:", df_imputed_X_minority_loras.shape)
print("Majority class:", df_imputed_X_majority.shape)

```

### Cell 16 — code (execution_count=None)
```python
# 4️⃣ Rebuild Balanced Set (Features + Labels)
X_loras = np.concatenate(
    (df_imputed_X_majority, df_imputed_X_minority_loras)
)

y_loras = np.concatenate(
    (
        np.ones(len(df_imputed_X_majority)),        # label = 1
        np.zeros(len(df_imputed_X_minority_loras))  # label = 0
    )
)

print("\nBalanced training set:")
print("X_loras:", X_loras.shape)
print("y_loras:", y_loras.shape)

```

### Cell 17 — markdown (execution_count=None)
```text
# **PCA to 9**
```

### Cell 18 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(X_loras)
```

### Cell 19 — markdown (execution_count=None)
```text
## Proposed OSEC
```

### Cell 20 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, y_loras, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 21 — code (execution_count=None)
```python
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

print("="*70)
print("STACKING CLASSIFIER WITH COMPREHENSIVE GRID SEARCH")
print("RECOMMENDED PARAMETER GRIDS - OPTIMIZED FOR BEST PERFORMANCE")
print("="*70)
print("\nNote: This will take significant time due to exhaustive search.")
print("Starting Grid Search for hyperparameter tuning...\n")

start_time = time.time()

# 1. Decision Tree - RECOMMENDED
print("1. Tuning Decision Tree...")
param_grid_tree = {
    'max_depth': [5],
    'max_features': ['sqrt', None],
    'min_samples_leaf': [2],
    'min_samples_split': [2],
    'criterion': ['entropy'],
    'splitter': ['best']
}
tree = DecisionTreeClassifier(random_state=42)
tree_grid = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
tree_grid.fit(X_train, y_train)
best_tree = tree_grid.best_estimator_
print(f"✓ Best Decision Tree params: {tree_grid.best_params_}")
print(f"✓ Best Decision Tree CV score: {tree_grid.best_score_:.4f}\n")

# 2. Logistic Regression - RECOMMENDED
print("2. Tuning Logistic Regression...")
param_grid_logistic = {
    'penalty': ['l2'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}
logistic = LogisticRegression(random_state=42, max_iter=2000)
logistic_grid = GridSearchCV(logistic, param_grid_logistic, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
logistic_grid.fit(X_train, y_train)
best_logistic = logistic_grid.best_estimator_
print(f"✓ Best Logistic Regression params: {logistic_grid.best_params_}")
print(f"✓ Best Logistic Regression CV score: {logistic_grid.best_score_:.4f}\n")

# 3. Bayesian Networks (Gaussian Naive Bayes) - RECOMMENDED
print("3. Tuning Bayesian Networks...")
param_grid_bayes = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
bayes = GaussianNB()
bayes_grid = GridSearchCV(bayes, param_grid_bayes, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
bayes_grid.fit(X_train, y_train)
best_bayes = bayes_grid.best_estimator_
print(f"✓ Best Bayesian Networks params: {bayes_grid.best_params_}")
print(f"✓ Best Bayesian Networks CV score: {bayes_grid.best_score_:.4f}\n")

print("="*70)

# 4. AdaBoost - RECOMMENDED (FIXED)
print("4. Tuning AdaBoost...")
param_grid_ab = {
    'n_estimators': [75, 100, 125],
    'learning_rate': [1.0, 0.1, 0.001, 0.0001],
    'algorithm': ['SAMME']  # FIXED: Changed from 'SAMME.R' to 'SAMME'
}
adaboost = AdaBoostClassifier(random_state=42)
ab_grid = GridSearchCV(adaboost, param_grid_ab, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
ab_grid.fit(X_train, y_train)
best_adaboost = ab_grid.best_estimator_
print(f"✓ Best AdaBoost params: {ab_grid.best_params_}")
print(f"✓ Best AdaBoost CV score: {ab_grid.best_score_:.4f}\n")

```

### Cell 22 — code (execution_count=None)
```python
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

print("="*70)

# 5. XGBoost Meta-learner - RECOMMENDED
print("5. Tuning XGBoost Meta-learner...")
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5]
}
xgb_meta = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_meta, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
print(f"✓ Best XGBoost params: {xgb_grid.best_params_}")
print(f"✓ Best XGBoost CV score: {xgb_grid.best_score_:.4f}\n")

grid_search_time = time.time() - start_time
print("="*70)
print(f"Grid Search completed in {grid_search_time/60:.2f} minutes!")
print("Building Stacking Classifier...")
print("="*70)

# Create the Stacking Classifier with tuned estimators
stacking_classifier = StackingClassifier(
    cv=5,
    estimators=[
        ('Bayesian Networks', best_bayes),
        ('AdaBoost', best_adaboost),
        ('Decision Tree', best_tree),
        ('Logistic Regression', best_logistic)
    ],
    final_estimator=best_xgb,
    n_jobs=-1
)

# Calibrate the Stacking Classifier
calibrated_stacking = CalibratedClassifierCV(stacking_classifier, method='sigmoid', cv=5, n_jobs=-1)

# Train the calibrated stacking classifier
print("\nTraining the Stacking Classifier with optimized hyperparameters...")
training_start = time.time()
calibrated_stacking.fit(X_train, y_train)
training_time = time.time() - training_start
print(f"Training completed in {training_time:.2f} seconds!")

```

### Cell 23 — code (execution_count=None)
```python
# Make predictions
print("\nMaking predictions on test set...")
y_pred = calibrated_stacking.predict(X_test)
y_pred_proba = calibrated_stacking.predict_proba(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # Use 'weighted' for multiclass
recall = recall_score(y_test, y_pred, average='binary')  # Use 'weighted' for multiclass
f1 = f1_score(y_test, y_pred, average='binary')  # Use 'weighted' for multiclass
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])  # For binary classification

# Display results
print("\n" + "="*70)
print("STACKING CLASSIFIER PERFORMANCE METRICS")
print("With RECOMMENDED Grid Search Hyperparameter Tuning")
print("Meta-learner: XGBoost")
print("="*70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print("="*70)

# Display classification report
print("\nDetailed Classification Report:")
print("-"*70)
print(classification_report(y_test, y_pred, digits=4))

# Display confusion matrix
print("\nConfusion Matrix:")
print("-"*70)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Summary of best parameters
print("\n" + "="*70)
print("BEST HYPERPARAMETERS SUMMARY")
print("="*70)
print("\n1. Decision Tree:")
for key, value in tree_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {tree_grid.best_score_:.4f}")

print("\n2. Logistic Regression:")
for key, value in logistic_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {logistic_grid.best_score_:.4f}")

print("\n3. Bayesian Networks:")
for key, value in bayes_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {bayes_grid.best_score_:.4f}")

print("\n4. AdaBoost:")
for key, value in ab_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {ab_grid.best_score_:.4f}")

print("\n5. XGBoost (Meta-learner):")
for key, value in xgb_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {xgb_grid.best_score_:.4f}")

print("="*70)
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time/60:.2f} minutes")
print("="*70)
```

### Cell 24 — markdown (execution_count=None)
```text
## 95% Confidence Intervals for AUC/F1
```

### Cell 25 — code (execution_count=None)
```python
# Universal Bootstrap CI Function (Reusable)
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np

def bootstrap_ci_model(
    model,
    X,
    y,
    n_bootstraps=1000,
    alpha=0.95,
    average='binary',
    model_name="Model"
):
    rng = np.random.RandomState(42)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": []
    }

    for _ in range(n_bootstraps):
        X_res, y_res = resample(X, y, random_state=rng)

        y_pred = model.predict(X_res)
        y_prob = model.predict_proba(X_res)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_res, y_pred))
        metrics["precision"].append(
            precision_score(y_res, y_pred, average=average, zero_division=0)
        )
        metrics["recall"].append(recall_score(y_res, y_pred, average=average))
        metrics["f1"].append(f1_score(y_res, y_pred, average=average))
        metrics["roc_auc"].append(roc_auc_score(y_res, y_prob))

    lower = (1 - alpha) / 2 * 100
    upper = (1 + alpha) / 2 * 100

    results = {}
    for metric, values in metrics.items():
        results[metric] = (
            np.percentile(values, lower),
            np.percentile(values, upper)
        )

    return results

```

### Cell 26 — code (execution_count=None)
```python
# 1️⃣ Store models (already tuned in your pipeline)
models = {
    "Naive Bayes": best_bayes,
    "Decision Tree": best_tree,
    "Logistic Regression": best_logistic,
    "AdaBoost": best_adaboost,
    "XGBoost": best_xgb,
    "Stacking": calibrated_stacking
}

#  2️⃣ Compute Bootstrap CI for each model
bootstrap_results_all = {}

print("\nRunning bootstrap (n=1000) for ALL models...\n")

for name, model in models.items():
    print(f"Bootstrapping {name}...")
    bootstrap_results_all[name] = bootstrap_ci_model(
        model=model,
        X=X_test,
        y=y_test,
        n_bootstraps=1000,
        alpha=0.95
    )

# Print Results (Side-by-side, Paper-ready)
print("\n" + "="*90)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (n = 1000)")
print("="*90)

for name, ci in bootstrap_results_all.items():
    print(f"\n{name}")
    print("-" * 90)
    print(f"Accuracy  CI: [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]")
    print(f"Precision CI: [{ci['precision'][0]:.4f}, {ci['precision'][1]:.4f}]")
    print(f"Recall    CI: [{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}]")
    print(f"F1-score  CI: [{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}]")
    print(f"ROC-AUC   CI: [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]")

```

### Cell 27 — code (execution_count=None)
```python
# Add point estimates automatically

def point_estimates(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob)
    }

print("\n" + "="*90)
print("POINT ESTIMATES + BOOTSTRAP CI")
print("="*90)

for name, model in models.items():
    pe = point_estimates(model, X_test, y_test)
    ci = bootstrap_results_all[name]

    print(f"\n{name}")
    print("-" * 90)
    for metric in pe:
        print(
            f"{metric.upper():9s}: {pe[metric]:.4f} "
            f"[{ci[metric][0]:.4f}, {ci[metric][1]:.4f}]"
        )

```

### Cell 28 — markdown (execution_count=None)
```text
## Paired Bootstrap Significance Test Stacking (ALL metrics)
```

### Cell 29 — code (execution_count=None)
```python
# 1️⃣ Metric dictionary (single source of truth)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

metric_fns = {
    "accuracy": lambda y, yp, yprob: accuracy_score(y, yp),
    "precision": lambda y, yp, yprob: precision_score(y, yp, zero_division=0),
    "recall": lambda y, yp, yprob: recall_score(y, yp),
    "f1": lambda y, yp, yprob: f1_score(y, yp),
    "roc_auc": lambda y, yp, yprob: roc_auc_score(y, yprob),
}

```

### Cell 30 — code (execution_count=None)
```python
# Paired Bootstrap Significance Test (ALL metrics)
# Bootstrap function
from sklearn.utils import resample
import numpy as np

def paired_bootstrap_all_metrics(
    model_1,
    model_2,
    X,
    y,
    metric_fns,
    n_bootstraps=1000
):
    diffs = {m: [] for m in metric_fns}

    for _ in range(n_bootstraps):
        X_res, y_res = resample(X, y)

        y1 = model_1.predict(X_res)
        y1p = model_1.predict_proba(X_res)[:, 1]

        y2 = model_2.predict(X_res)
        y2p = model_2.predict_proba(X_res)[:, 1]

        for name, fn in metric_fns.items():
            m1 = fn(y_res, y1, y1p)
            m2 = fn(y_res, y2, y2p)
            diffs[name].append(m1 - m2)

    results = {}
    for m, values in diffs.items():
        values = np.array(values)

        p_value = min(1.0, 2 * min(
            np.mean(values <= 0),
            np.mean(values >= 0)
            ))
        results[m] = {
            "mean_diff": np.mean(values),
            "p_value": p_value
        }

    return results

```

### Cell 31 — code (execution_count=None)
```python
# Apply (Stacking vs Base models)
base_models = {
    "Naive Bayes": best_bayes,
    "Decision Tree": best_tree,
    "Logistic Regression": best_logistic,
    "AdaBoost": best_adaboost,
    "XGBoost": best_xgb
}

bootstrap_tests = {}

print("\nPaired Bootstrap Tests (Stacking vs Base Models)\n")

for name, model in base_models.items():
    print(f"Bootstrap: Stacking vs {name}")

    bootstrap_tests[name] = paired_bootstrap_all_metrics(
        calibrated_stacking,
        model,
        X_test,
        y_test,
        metric_fns,
        n_bootstraps=1000
    )

```

### Cell 32 — code (execution_count=None)
```python
# Bootstrap results
print("\n" + "="*100)
print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS (n = 1000)")
print("="*100)

for model, metrics in bootstrap_tests.items():
    print(f"\nStacking vs {model}")
    print("-"*100)
    for m, res in metrics.items():
        print(
            f"{m.upper():9s} | Δ = {res['mean_diff']:+.4f} | p = {res['p_value']:.6f}"
        )

```

### Cell 33 — markdown (execution_count=None)
```text
## **Wilcoxon (Stacking vs Base models)**
```

### Cell 34 — code (execution_count=None)
```python
# Apply Wilcoxon (Stacking vs Base models)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon

def cv_scores_all_metrics(model, X, y, metric_fns, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {m: [] for m in metric_fns}

    for tr, val in skf.split(X, y):
        X_tr, X_val = X[tr], X[val]
        y_tr, y_val = y[tr], y[val]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        for name, fn in metric_fns.items():
            scores[name].append(fn(y_val, y_pred, y_prob))

    return scores

```

### Cell 35 — code (execution_count=None)
```python
# Wilcoxon results
wilcoxon_tests = {}

print("\nWilcoxon Tests Across CV Folds\n")

for name, model in base_models.items():
    print(f"Wilcoxon: Stacking vs {name}")

    stack_scores = cv_scores_all_metrics(
        calibrated_stacking, X_train, y_train, metric_fns
    )
    base_scores = cv_scores_all_metrics(
        model, X_train, y_train, metric_fns
    )

    wilcoxon_tests[name] = {}

    for metric in metric_fns:
        stat, p = wilcoxon(
            stack_scores[metric],
            base_scores[metric]
        )
        wilcoxon_tests[name][metric] = {
            "stat": stat,
            "p_value": p
        }

```

### Cell 36 — code (execution_count=None)
```python
print("\n" + "="*100)
print("WILCOXON SIGNED-RANK TESTS (5-Fold CV)")
print("="*100)

for model, metrics in wilcoxon_tests.items():
    print(f"\nStacking vs {model}")
    print("-"*100)
    for m, res in metrics.items():
        print(
            f"{m.upper():9s} | stat = {res['stat']:.4f} | p = {res['p_value']:.6f}"
        )
```

### Cell 37 — markdown (execution_count=None)
```text
## Proposed OVEC
```

### Cell 38 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# --- Random Search for Base Models ---

# 1. Logistic Regression
param_log = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['lbfgs']
}
log_rs = RandomizedSearchCV(LogisticRegression(), param_log, n_iter=10, cv=5, scoring='accuracy', random_state=42)
log_rs.fit(X_train, y_train)
logistic_best = log_rs.best_estimator_

# 2. Random Forest
param_rf = {
    'max_depth': [None],
    'n_estimators': [2],
    'criterion': ['entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train, y_train)
rf_best = rf_rs.best_estimator_

# 3. KNN
param_knn = {
    'n_neighbors': [5],
    'weights': ['distance'],
    'metric': ['manhattan']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train, y_train)
knn_best = knn_rs.best_estimator_

# 4. SVM
param_svm = {
    'C': [100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train, y_train)
svm_best = svm_rs.best_estimator_

```

### Cell 39 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, brier_score_loss, average_precision_score,
    precision_recall_curve
)
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Create the Stacking Classifier with tuned estimators
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)

# Combine tuned models into VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ('knn', knn_best),
        ('rf', rf_best),
        ('logistic', logistic_best),
        ('svm', svm_best)
    ],
    voting='soft',
    weights=[1, 2, 1, 2],  # Example weights
    n_jobs=-1
)

# Fit the model
#voting_model.fit(X_train, y_train)

# ----------------------------------------------------------
# 3. Apply Platt Scaling (Sigmoid Calibration)
# ----------------------------------------------------------

platt_calibrated_Voting_model = CalibratedClassifierCV(
    estimator=voting_model,
    method='sigmoid',
    cv=cv,
    n_jobs=-1
)

platt_calibrated_Voting_model.fit(X_train, y_train)
```

### Cell 40 — markdown (execution_count=None)
```text
## Paired Bootstrap Significance Test Voting (ALL metrics)
```

### Cell 41 — code (execution_count=None)
```python
# 1️⃣ Metric dictionary (single source of truth)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }
```

### Cell 42 — code (execution_count=None)
```python
# ✅ 2. Revised Paired Bootstrap Significance Test
import numpy as np

def paired_bootstrap_test(
    y_true,
    y_pred_a, y_proba_a,
    y_pred_b, y_proba_b,
    n_bootstrap=1000,
    alpha=0.05,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    metric_names = compute_metrics(
        y_true, y_pred_a, y_proba_a
    ).keys()

    metric_diffs = {m: [] for m in metric_names}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)

        metrics_a = compute_metrics(
            y_true[idx], y_pred_a[idx], y_proba_a[idx]
        )
        metrics_b = compute_metrics(
            y_true[idx], y_pred_b[idx], y_proba_b[idx]
        )

        for m in metric_names:
            metric_diffs[m].append(metrics_b[m] - metrics_a[m])

    results = {}
    for m, diffs in metric_diffs.items():
        diffs = np.array(diffs)

        lower = np.percentile(diffs, 100 * alpha / 2)
        upper = np.percentile(diffs, 100 * (1 - alpha / 2))

        p_value = 2 * min(
            np.mean(diffs <= 0),
            np.mean(diffs >= 0)
        )

        results[m] = {
            "mean_diff": diffs.mean(),
            "ci_95": (lower, upper),
            "p_value": p_value,
            "significant": p_value < alpha
        }

    return results

```

### Cell 43 — code (execution_count=None)
```python
base_models = {
    "logistic": logistic_best,
    "randomforest": rf_best,
    "knn": knn_best,
    "svm": svm_best
}

# Predictions of calibrated voting model (fixed reference)
y_pred_vote = platt_calibrated_Voting_model.predict(X_test)
y_proba_vote = platt_calibrated_Voting_model.predict_proba(X_test)[:, 1]

# Ensure numpy array
y_test_np = y_test.values if hasattr(y_test, "values") else y_test
```

### Cell 44 — code (execution_count=None)
```python
# ==========================================================
# 5. RUN PAIRED BOOTSTRAP FOR ALL BASE MODELS
# ==========================================================

for model_name, model in base_models.items():

    # Base model predictions
    y_pred_base = model.predict(X_test)
    y_proba_base = model.predict_proba(X_test)[:, 1]

    results = paired_bootstrap_test(
        y_test,
        y_pred_base, y_proba_base,
        y_pred_vote, y_proba_vote
    )

    print("\n" + "=" * 60)
    print(f"{model_name.upper()}  vs  CALIBRATED VOTING MODEL")
    print("=" * 60)

    for m, res in results.items():
        print(
            f"{m.upper():9s} | Δ = {res['mean_diff']:+.4f} | "
            f"p = {res['p_value']:.6f}"
        )

```

### Cell 45 — markdown (execution_count=None)
```text
## Prepare for FIVE STATE Test Data
```

### Cell 46 — code (execution_count=None)
```python
data = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data
data.head()
```

### Cell 47 — code (execution_count=None)
```python
print(data.columns.tolist())

```

### Cell 48 — code (execution_count=None)
```python
# Preprocessing
data = data.dropna()  # Drop rows with missing values

# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'

# Convert 'Sex' to 0 and 1 (e.g., Male = 1, Female = 0)
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})

# Encode 'ChestPainType' (use LabelEncoder for multi-class categorical variables)
chest_pain_encoder = LabelEncoder()
data['ChestPainType'] = chest_pain_encoder.fit_transform(data['ChestPainType'])

# Encode 'RestingECG'
resting_ecg_encoder = LabelEncoder()
data['RestingECG'] = resting_ecg_encoder.fit_transform(data['RestingECG'])

# Convert 'ExerciseAngina' to binary (e.g., Y = 1, N = 0)
data['ExerciseAngina'] = data['ExerciseAngina'].map({'Y': 1, 'N': 0})
data['ST_Slope'] = data['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': 0})

#first 5 values of data
data.head()
```

### Cell 49 — code (execution_count=None)
```python
X = data.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data['HeartDisease']
print(data)
print("X",X)
print("y",y)

#Target Variables Values
data.HeartDisease.value_counts()
```

### Cell 50 — code (execution_count=None)
```python
import numpy as np
import pandas as pd

# Assume 'data' is your DataFrame with shape (918, 12)
# For the purpose of this example, we will generate a sample DataFrame
# data = pd.read_csv('your_dataset.csv')

# Define percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Detect outliers based on percentiles for each column
outliers = pd.DataFrame()

# Iterate over each feature in the dataset
for column in data.columns:
    # Calculate the lower and upper percentile values for the current column
    lower_bound = np.percentile(data[column], lower_percentile)
    upper_bound = np.percentile(data[column], upper_percentile)

    # Flag outliers in this column
    outliers[column] = (data[column] < lower_bound) | (data[column] > upper_bound)

# Combine the outliers across all columns
# An outlier is detected if it's flagged in any feature
outliers['is_outlier'] = outliers.any(axis=1)

# Filter the rows that are flagged as outliers
data_outliers = data[outliers['is_outlier']]
data_no_outliers = data[~outliers['is_outlier']]

# Output the result
print("Number of outliers detected:", data_outliers.shape[0])
print("Data with outliers removed:", data_no_outliers.shape)

#Target Variables Values
data_outliers.HeartDisease.value_counts()

X = data_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data_no_outliers['HeartDisease']


#Target Variables Values
data_no_outliers.HeartDisease.value_counts()
X.shape
```

### Cell 51 — code (execution_count=None)
```python
import pandas as pd
import numpy as np

# Define the lower and upper percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Identify numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns

# Function to mark outliers as NaN
def mark_outliers_as_nan(df, cols, lower_perc, upper_perc):
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    for col in cols:
        lower_bound = np.percentile(df_copy[col], lower_perc)
        upper_bound = np.percentile(df_copy[col], upper_perc)
        df_copy[col] = np.where((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound), np.nan, df_copy[col])
    return df_copy  # Return the modified DataFrame

# Store the modified DataFrame
data_with_nans = mark_outliers_as_nan(data, numerical_cols, lower_percentile, upper_percentile)

# Check the result
print("Data after marking outliers as NaN:")
print(data_with_nans.isnull().sum())
```

### Cell 52 — code (execution_count=None)
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Assuming your Outliers dataset after removing outliers
# The label column is assumed to be named 'HeartDisease'

# Split data into features and labels
X_Outliers = data_with_nans.drop('HeartDisease', axis=1)  # Features
y_Outliers = data_with_nans['HeartDisease']               # Labels

# Separate into heart disease and normal groups
df_heart_disease = data_with_nans[data_with_nans['HeartDisease'] == 1].copy()
df_normal = data_with_nans[data_with_nans['HeartDisease'] == 0].copy()

# Verify data before imputation
print("Shape of Heart Disease group before numerical imputation: ", df_heart_disease.shape)
print("Shape of Normal group before numerical imputation: ", df_normal.shape)

# Impute Numerical Columns
numerical_cols = X_Outliers.select_dtypes(include=[np.number]).columns
median_imputer = SimpleImputer(strategy='median')

df_heart_disease[numerical_cols] = median_imputer.fit_transform(df_heart_disease[numerical_cols])
df_normal[numerical_cols] = median_imputer.fit_transform(df_normal[numerical_cols])

print("Shape of Heart Disease group after numerical imputation: ", df_heart_disease.shape)
print("Shape of Normal group after numerical imputation: ", df_normal.shape)

# Combine both groups back together after imputation
df_imputed = pd.concat([df_heart_disease, df_normal], ignore_index=True)

print("Final Imputed DataFrame: \n", df_imputed.head())
print("Missing values after imputation: \n", df_imputed.isnull().sum())
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 53 — code (execution_count=None)
```python
# 1️⃣ Separate Classes from TRAINING DATA ONLY
# Install LoRAS package
!pip install loras

# Import necessary libraries
import loras
import numpy as np
import pandas as pd

# Majority and minority from TRAINING SET ONLY
df_imputed_X_majority = df_imputed_X[df_imputed_y == 1]   # e.g., HeartDisease = 1
df_imputed_X_minority = df_imputed_X[df_imputed_y == 0]   # e.g., HeartDisease = 0

print("Before LoRAS:")
print("Majority class:", df_imputed_X_majority.shape)
print("Minority class:", df_imputed_X_minority.shape)

```

### Cell 54 — code (execution_count=None)
```python
# 2️⃣ Apply LoRAS
df_imputed_X_minority_loras = loras.fit_resample( df_imputed_X_majority, df_imputed_X_minority)
print("After LoRAS:")
print("Minority class:", df_imputed_X_minority_loras.shape)
print("Majority class:", df_imputed_X_majority.shape)
```

### Cell 55 — code (execution_count=None)
```python
import numpy as np

# 3️⃣ Downsample If LoRAS Overshoots (Optional but Recommended)
if df_imputed_X_minority_loras.shape[0] > df_imputed_X_majority.shape[0]:
    np.random.seed(42)

    selected_indices = np.random.choice(
        df_imputed_X_minority_loras.shape[0],
        size=df_imputed_X_majority.shape[0],
        replace=False
    )

    df_imputed_X_minority_loras = df_imputed_X_minority_loras[selected_indices]

print("\nAfter LoRAS + Downsampling:")
print("Minority class:", df_imputed_X_minority_loras.shape)
print("Majority class:", df_imputed_X_majority.shape)

```

### Cell 56 — code (execution_count=None)
```python
# 4️⃣ Rebuild Balanced Set (Features + Labels)
X_loras = np.concatenate(
    (df_imputed_X_majority, df_imputed_X_minority_loras)
)

y_loras = np.concatenate(
    (
        np.ones(len(df_imputed_X_majority)),        # label = 1
        np.zeros(len(df_imputed_X_minority_loras))  # label = 0
    )
)

print("\nBalanced training set:")
print("X_loras:", X_loras.shape)
print("y_loras:", y_loras.shape)

```

### Cell 57 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(X_loras)
```

### Cell 58 — code (execution_count=None)
```python
X_test_five = X_oversampled_LoRAS_features_pca9
y_test_five = y_loras

```

### Cell 59 — markdown (execution_count=None)
```text
# Proposed OSEC on FIVE States Processed data
```

### Cell 60 — code (execution_count=None)
```python
#
# Make predictions by Proposed OSEC
#
print("\nMaking predictions on test_five set...")
y_pred = calibrated_stacking.predict(X_test_five)
y_pred_proba = calibrated_stacking.predict_proba(X_test_five)

# Calculate performance metrics
accuracy = accuracy_score(y_test_five, y_pred)
precision = precision_score(y_test_five, y_pred, average='binary')  # Use 'weighted' for multiclass
recall = recall_score(y_test_five, y_pred, average='binary')  # Use 'weighted' for multiclass
f1 = f1_score(y_test_five, y_pred, average='binary')  # Use 'weighted' for multiclass
roc_auc = roc_auc_score(y_test_five, y_pred_proba[:, 1])  # For binary classification

# Display results
print("\n" + "="*70)
print("STACKING CLASSIFIER PERFORMANCE METRICS")
print("With RECOMMENDED Grid Search Hyperparameter Tuning")
print("Meta-learner: XGBoost")
print("="*70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print("="*70)

# Display classification report
print("\nDetailed Classification Report:")
print("-"*70)
print(classification_report(y_test_five, y_pred, digits=4))

# Display confusion matrix
print("\nConfusion Matrix:")
print("-"*70)
cm = confusion_matrix(y_test_five, y_pred)
print(cm)
print()

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Summary of best parameters
print("\n" + "="*70)
print("BEST HYPERPARAMETERS SUMMARY")
print("="*70)
print("\n1. Decision Tree:")
for key, value in tree_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {tree_grid.best_score_:.4f}")

print("\n2. Logistic Regression:")
for key, value in logistic_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {logistic_grid.best_score_:.4f}")

print("\n3. Bayesian Networks:")
for key, value in bayes_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {bayes_grid.best_score_:.4f}")

print("\n4. AdaBoost:")
for key, value in ab_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {ab_grid.best_score_:.4f}")

print("\n5. XGBoost (Meta-learner):")
for key, value in xgb_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {xgb_grid.best_score_:.4f}")

print("="*70)
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time/60:.2f} minutes")
print("="*70)
```

### Cell 61 — markdown (execution_count=None)
```text

```

### Cell 62 — markdown (execution_count=None)
```text
# Proposed OSEC on FIVE States Original data
```

### Cell 63 — code (execution_count=None)
```python
X_five = X
y_five = y
print("X",X)
print("y",y)
```

### Cell 64 — code (execution_count=None)
```python
#
# Make predictions by Proposed OSEC
#
print("\nMaking predictions on test_five set...")
y_pred = calibrated_stacking.predict(X_five)
y_pred_proba = calibrated_stacking.predict_proba(X_five)

# Calculate performance metrics
accuracy = accuracy_score(y_five, y_pred)
precision = precision_score(y_five, y_pred, average='binary')  # Use 'weighted' for multiclass
recall = recall_score(y_five, y_pred, average='binary')  # Use 'weighted' for multiclass
f1 = f1_score(y_five, y_pred, average='binary')  # Use 'weighted' for multiclass
roc_auc = roc_auc_score(y_five, y_pred_proba[:, 1])  # For binary classification

# Display results
print("\n" + "="*70)
print("STACKING CLASSIFIER PERFORMANCE METRICS")
print("With RECOMMENDED Grid Search Hyperparameter Tuning")
print("Meta-learner: XGBoost")
print("="*70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print("="*70)

# Display classification report
print("\nDetailed Classification Report:")
print("-"*70)
print(classification_report(y_five, y_pred, digits=4))

# Display confusion matrix
print("\nConfusion Matrix:")
print("-"*70)
cm = confusion_matrix(y_five, y_pred)
print(cm)
print()

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# Summary of best parameters
print("\n" + "="*70)
print("BEST HYPERPARAMETERS SUMMARY")
print("="*70)
print("\n1. Decision Tree:")
for key, value in tree_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {tree_grid.best_score_:.4f}")

print("\n2. Logistic Regression:")
for key, value in logistic_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {logistic_grid.best_score_:.4f}")

print("\n3. Bayesian Networks:")
for key, value in bayes_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {bayes_grid.best_score_:.4f}")

print("\n4. AdaBoost:")
for key, value in ab_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {ab_grid.best_score_:.4f}")

print("\n5. XGBoost (Meta-learner):")
for key, value in xgb_grid.best_params_.items():
    print(f"   {key}: {value}")
print(f"   CV Score: {xgb_grid.best_score_:.4f}")

print("="*70)
total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time/60:.2f} minutes")
print("="*70)
```

### Cell 65 — markdown (execution_count=None)
```text
## Proposed OVEC on FIVE States Processed data
```

### Cell 66 — code (execution_count=None)
```python
# Make predictions
voting_predictions = voting_model.predict(X_test_five)

# Evaluate Metrics
#train_accuracy = voting_model.score(X_train, y_train)
test_five_accuracy = accuracy_score(y_test_five, voting_predictions)
precision = precision_score(y_test_five, voting_predictions)
recall = recall_score(y_test_five, voting_predictions)
f1 = f1_score(y_test_five, voting_predictions)

# AUC-ROC Curve
voting_probabilities = voting_model.predict_proba(X_test_five)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_five, voting_probabilities)
auc_score = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Voting Classifier AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Display all metrics
print("========== Model Performance with Weighted Voting ==========")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test_fiveing Accuracy : {test_five_accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"AUC-ROC           : {auc_score:.4f}")
print("===========================================================")

```