# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\mendeley_results\MENDELEY Results\Mendeley_Clinical_Utility+Dispersion+Statistical_Tests.ipynb`
- SHA-256: `aa1649014e0ed66fbea56bd6f544b9d4b2d32a3742ba6cbd7e40eef0e7e3f242`
- Cells: 198
- Saved outputs: 167

## Imports
- `google.colab`
- `imblearn.over_sampling`
- `loras`
- `matplotlib.pyplot`
- `numpy`
- `os`
- `pandas`
- `random`
- `scipy.stats`
- `seaborn`
- `sklearn.calibration`
- `sklearn.decomposition`
- `sklearn.ensemble`
- `sklearn.impute`
- `sklearn.linear_model`
- `sklearn.metrics`
- `sklearn.model_selection`
- `sklearn.naive_bayes`
- `sklearn.neighbors`
- `sklearn.pipeline`
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
- OVEC
- RandomizedSearchCV
- SMOTE
- StackingClassifier
- VotingClassifier
- bootstrap
- train_test_split
- wilcoxon

## Risk flags
- **credential_or_mount**: cells [1, 78, 91, 99, 102, 140, 146, 150, 153, 185]
- **notebook_shell**: cells [22, 174]

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
mendeley = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv')

#first 5 values of data
mendeley.head()

```

### Cell 4 — code (execution_count=None)
```python
# Check the result
print("Data as NaN:")
print(mendeley.isnull().sum())
print("Data Shape:", mendeley.shape)
```

### Cell 5 — code (execution_count=None)
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Cell 6 — code (execution_count=None)
```python
# Rename Mendeley Columns
rename_dict = {
    "age": "Age",
    "gender": "Sex",
    "chestpain": "ChestPainType",
    "restingBP": "RestingBP",
    "serumcholestrol": "Cholesterol",
    "fastingbloodsugar": "FastingBS",
    "restingrelectro": "RestingECG",
    "maxheartrate": "MaxHR",
    "exerciseangia": "ExerciseAngina",
    "oldpeak": "Oldpeak",
    "slope": "ST_Slope",
    "target": "HeartDisease"
}

mendeley.rename(columns=rename_dict, inplace=True)

```

### Cell 7 — code (execution_count=None)
```python
# Drop Unwanted Columns
mendeley.drop(columns=["patientid", "noofmajorvessels"], inplace=True, errors="ignore")
mendeley.shape

```

### Cell 8 — code (execution_count=None)
```python
print("Mendeley ST_Slope:", sorted(mendeley["ST_Slope"].unique()))
```

### Cell 9 — code (execution_count=None)
```python
# Step 1️⃣ Remove ST_Slope == 0
mendeley= mendeley[mendeley["ST_Slope"] != 0].reset_index(drop=True)
print("Mendeley ST_Slope:", sorted(mendeley["ST_Slope"].unique()))
```

### Cell 10 — code (execution_count=None)
```python
mendeley["ST_Slope"] = mendeley["ST_Slope"] - 1
print("Mendeley ST_Slope:", sorted(mendeley["ST_Slope"].unique()))
```

### Cell 11 — markdown (execution_count=None)
```text
# **Step 1: Remove Label from Dataset
```

### Cell 12 — code (execution_count=None)
```python
X = mendeley.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = mendeley['HeartDisease']
print(mendeley)
print("X",X)
print("y",y)

#target Variables Values
mendeley.HeartDisease.value_counts()
```

### Cell 13 — markdown (execution_count=None)
```text
# **Anamoly/ Outllier Dection by Percentile Method**
```

### Cell 14 — code (execution_count=None)
```python
import numpy as np
import pandas as pd

# Assume 'mendeley' is your mendeleyFrame with shape (918, 12)
# For the purpose of this example, we will generate a sample mendeleyFrame
# mendeley = pd.read_csv('your_mendeleyset.csv')

# Define percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Detect outliers based on percentiles for each column
outliers = pd.DataFrame()

# Iterate over each feature in the mendeleyset
for column in mendeley.columns:
    # Calculate the lower and upper percentile values for the current column
    lower_bound = np.percentile(mendeley[column], lower_percentile)
    upper_bound = np.percentile(mendeley[column], upper_percentile)

    # Flag outliers in this column
    outliers[column] = (mendeley[column] < lower_bound) | (mendeley[column] > upper_bound)

# Combine the outliers across all columns
# An outlier is detected if it's flagged in any feature
outliers['is_outlier'] = outliers.any(axis=1)

# Filter the rows that are flagged as outliers
mendeley_outliers = mendeley[outliers['is_outlier']]
mendeley_no_outliers = mendeley[~outliers['is_outlier']]

# Output the result
print("Number of outliers detected:", mendeley_outliers.shape[0])
print("mendeley with outliers removed:", mendeley_no_outliers.shape)

#Target Variables Values
mendeley_outliers.HeartDisease.value_counts()
```

### Cell 15 — code (execution_count=None)
```python
X = mendeley_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'HeartDisease' is the label column
y = mendeley_no_outliers['HeartDisease']


#HeartDisease Variables Values
mendeley_no_outliers.HeartDisease.value_counts()
X.shape
```

### Cell 16 — code (execution_count=None)
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Count the occurrences of 'HeartDisease' for both outliers and non-outliers
outliers_heart_disease_counts = mendeley_outliers['HeartDisease'].value_counts().sort_index()
non_outliers_heart_disease_counts = mendeley_no_outliers['HeartDisease'].value_counts().sort_index()

# Create a figure and subplots
plt.figure(figsize=(12, 6))

# Colors for the bars
colors_outliers = ['blue', 'green']  # 0: No Heart Disease, 1: Heart Disease
colors_non_outliers = ['blue', 'green']  # Same color scheme

# Plot for outliers
plt.subplot(1, 2, 1)
sns.barplot(x=outliers_heart_disease_counts.index,
            y=outliers_heart_disease_counts.values,
            palette=colors_outliers)
plt.title('Heart Disease Distribution among Outliers', fontsize=14)
plt.xlabel('Heart Disease (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Annotate the bars with counts for outliers
for i, value in enumerate(outliers_heart_disease_counts.values):
    plt.text(i, value + 1, str(value), ha='center', fontsize=12, color='black')

# Plot for non-outliers
plt.subplot(1, 2, 2)
sns.barplot(x=non_outliers_heart_disease_counts.index,
            y=non_outliers_heart_disease_counts.values,
            palette=colors_non_outliers)
plt.title('Heart Disease Distribution among Non-Outliers', fontsize=14)
plt.xlabel('Heart Disease (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Annotate the bars with counts for non-outliers
for i, value in enumerate(non_outliers_heart_disease_counts.values):
    plt.text(i, value + 1, str(value), ha='center', fontsize=12, color='black')

# Adjust the layout and show the plots
plt.tight_layout()
plt.show()

```

### Cell 17 — markdown (execution_count=None)
```text
# **Dealing Outliers with Data Imputation Methods **
```

### Cell 18 — code (execution_count=None)
```python
import pandas as pd
import numpy as np

# Define the lower and upper percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Identify numerical columns
numerical_cols = mendeley.select_dtypes(include=[np.number]).columns

# Function to mark outliers as NaN
def mark_outliers_as_nan(df, cols, lower_perc, upper_perc):
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    for col in cols:
        lower_bound = np.percentile(df_copy[col], lower_perc)
        upper_bound = np.percentile(df_copy[col], upper_perc)
        df_copy[col] = np.where((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound), np.nan, df_copy[col])
    return df_copy  # Return the modified DataFrame

# Store the modified DataFrame
data_with_nans = mark_outliers_as_nan(mendeley, numerical_cols, lower_percentile, upper_percentile)

# Check the result
print("Data after marking outliers as NaN:")
print(data_with_nans.isnull().sum())
```

### Cell 19 — code (execution_count=None)
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
```

### Cell 20 — code (execution_count=None)
```python
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'HeartDisease' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 21 — markdown (execution_count=None)
```text


# Step 2: Apply Oversampling Technoques/ Data Balancing Techniques
LORAS

```

### Cell 22 — code (execution_count=None)
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

### Cell 23 — code (execution_count=None)
```python
# 2️⃣ Apply LoRAS
df_imputed_X_minority_loras = loras.fit_resample( df_imputed_X_majority, df_imputed_X_minority)
print("After LoRAS:")
print("Minority class:", df_imputed_X_minority_loras.shape)
print("Majority class:", df_imputed_X_majority.shape)
```

### Cell 24 — code (execution_count=None)
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

### Cell 25 — code (execution_count=None)
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

### Cell 26 — markdown (execution_count=None)
```text
# **PCA to 9**
```

### Cell 27 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(X_loras)
```

### Cell 28 — markdown (execution_count=None)
```text
# **Daza Stacking 1A**
```

### Cell 29 — code (execution_count=None)
```python
# Apply oversampling using SMOTE to balance the dataset
smote = SMOTE()
X_daza, y_daza = smote.fit_resample(X, y)
```

### Cell 30 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_smote_features_pca9 = pca.fit_transform(X_daza)
```

### Cell 31 — code (execution_count=None)
```python
X_train_daza, X_test_daza, y_train_daza, y_test_daza = train_test_split(X_oversampled_smote_features_pca9 , y_daza, test_size=0.25, random_state=42, shuffle=True)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train_daza.shape}')
print(f'Test features shape: {X_test_daza.shape}')
print(f'Train labels shape: {y_train_daza.shape}')
print(f'Test labels shape: {y_test_daza.shape}')
```

### Cell 32 — code (execution_count=None)
```python
# D Random Forest
param_grid_rf = {
    'n_estimators': [25, 50, 100],
    'max_depth': [5, None],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [5, 10, None],
    'criterion': ['gini', 'entropy']
}

#  A Decision Tree
param_grid_tree = {
    'max_depth': [2, 4, 5, None],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

# B Gradient Descent
param_grid_gradien = {
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [20, 50, 100, 150]
}

# G Bayesian Networks
param_grid_bayes = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

# E Logistic Regression
param_grid_logistic = {
    'penalty': ['l1', 'l2', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']
}

# Perform grid search on the individual models

# A Decision Tree Grid Search (Base 1 Classifier)
tree = DecisionTreeClassifier()
grid_search_tree_daza = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree_daza.fit(X_train_daza, y_train_daza)
best_tree_daza = grid_search_tree_daza.best_estimator_
# Output the best parameters
print("Best Parameters of Decision Tree Grid Search (Base 1 Classifier):", best_tree_daza)

# B Gradient Boosting Grid Search (Base 2 Classifier)
gradien = GradientBoostingClassifier()
grid_search_gradien_daza = GridSearchCV(gradien, param_grid_gradien, cv=5, scoring='accuracy')
grid_search_gradien_daza.fit(X_train_daza, y_train_daza)
best_gradien_daza = grid_search_gradien_daza.best_estimator_
# Output the best parameters
print("Best Parameters of Gradient Boosting Grid Search (Base 2 Classifier):", best_gradien_daza)

# D Random Forest Grid Search (Base 3 Classifier)
rf = RandomForestClassifier()
grid_search_rf_daza = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf_daza.fit(X_train_daza, y_train_daza)
best_rf_daza = grid_search_rf_daza.best_estimator_
# Output the best parameters
print("Best Parameters of Random Forest Grid Search (Base 3 Classifier):", best_rf_daza)

# E Logistic Regression Grid Search (Meta Classifier)
logistic = LogisticRegression()
grid_search_logistic_daza = GridSearchCV(logistic, param_grid_logistic, cv=5, scoring='accuracy')
grid_search_logistic_daza.fit(X_train_daza, y_train_daza)
best_logistic_daza = grid_search_logistic_daza.best_estimator_
# Output the best parameters
print("Best Parameters of Logistic Regression Grid Search (Meta Classifier):", best_logistic_daza)

# G Naive Bayes Grid Search (Base 4 Classifier)
bayes = GaussianNB()
grid_search_bayes_daza = GridSearchCV(bayes, param_grid_bayes, cv=5, scoring='accuracy')
grid_search_bayes_daza.fit(X_train_daza, y_train_daza)
best_bayes_daza = grid_search_bayes_daza.best_estimator_
# Output the best parameters
print("Best Parameters of Naive Bayes Grid Search (Base 4 Classifier):", best_bayes_daza)
```

### Cell 33 — code (execution_count=None)
```python
# ============================================================
# Full Reproducible Stacking Classifier Pipeline
# ============================================================

import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

# ============================================================
# 1️⃣ Set Global Random Seeds
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ============================================================
# 2️⃣ Ensure Random States in All Estimators
# ============================================================
best_rf_daza.random_state = SEED
best_tree_daza.random_state = SEED
best_gradien_daza.random_state = SEED
best_bayes_daza.random_state = SEED
best_logistic_daza.random_state = SEED

# ============================================================
# 3. Ensure All Models Use the Same Random State
# ============================================================
for mdl in [
    best_rf_daza, best_tree_daza, best_gradien_daza,
    best_bayes_daza, best_logistic_daza
]:
    if hasattr(mdl, "random_state"):
        mdl.random_state = SEED

```

### Cell 34 — code (execution_count=None)
```python
# ============================================================
# 4. Logistic Regression (Clean Pipeline)
# ============================================================
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=SEED))
])

logreg_pipeline.fit(X_train_daza, y_train_daza)
logreg_proba = logreg_pipeline.predict_proba(X_test_daza)[:, 1]

```

### Cell 35 — code (execution_count=None)
```python
# ============================================================
# 5. Stacking Classifier (Correct Training)
# ============================================================
stacking_classifier_daza = StackingClassifier(
    estimators=[
        ('rf', best_rf_daza),
        ('tree', best_tree_daza),
        ('gradien', best_gradien_daza),
        ('bayes', best_bayes_daza)
    ],
    final_estimator=best_logistic_daza,
    cv=5,
    passthrough=False
)

# Calibrate the stacking model properly (fit ONCE)
calibrated_stacking_daza = CalibratedClassifierCV(
    stacking_classifier_daza, method="sigmoid", cv=5
)

calibrated_stacking_daza.fit(X_train_daza, y_train_daza)
stacking_proba = calibrated_stacking_daza.predict_proba(X_test_daza)[:, 1]
```

### Cell 36 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt

# Predict on the test data
y_pred_daza = calibrated_stacking_daza.predict(X_test_daza)
y_pred_proba_daza = calibrated_stacking_daza.predict_proba(X_test_daza)[:, 1]

# Evaluate the model
accuracy_daza = accuracy_score(y_test_daza, y_pred_daza)
precision_daza = precision_score(y_test_daza, y_pred_daza)
recall_daza = recall_score(y_test_daza, y_pred_daza)
f1_daza = f1_score(y_test_daza, y_pred_daza)
auc_daza = roc_auc_score(y_test_daza, y_pred_proba_daza)

# Print evaluation results
print(f'Accuracy_daza: {accuracy_daza:.4f}')
print(f'Precision_daza: {precision_daza:.4f}')
print(f'Recall_daza: {recall_daza:.4f}')
print(f'F1-score_daza: {f1_daza:.4f}')
print(f'AUC-ROC_daza: {auc_daza:.4f}')

# Plot ROC Curve
fpr_daza, tpr_daza, thresholds_daza = roc_curve(y_test_daza, y_pred_proba_daza)
plt.figure()
plt.plot(fpr_daza, tpr_daza, color='blue', label=f'ROC curve (area_daza = {auc_daza:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

```

### Cell 37 — code (execution_count=None)
```python
len(y_test_daza) == 254
```

### Cell 38 — markdown (execution_count=None)
```text
## Proposed OSEC
```

### Cell 39 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, y_loras, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 40 — code (execution_count=None)
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

### Cell 41 — code (execution_count=None)
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    print("="*70)
    print(f"PERFORMANCE METRICS — {model_name}")
    print("="*70)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities (for ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)
        if y_test_proba.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
    else:
        roc_auc = "Not Available"

    # Metrics
    print("\nTest Metrics:")
    print(f" Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f" Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f" Recall   : {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f" F1-score : {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f" ROC-AUC  : {roc_auc}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

evaluate_model(best_tree, X_test, y_test, "Decision Tree")
evaluate_model(best_logistic, X_test, y_test, "Logistic Regression")
evaluate_model(best_bayes, X_test, y_test, "Gaussian Naive Bayes")
evaluate_model(best_adaboost, X_test, y_test, "AdaBoost")


```

### Cell 42 — code (execution_count=None)
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
    "n_estimators": [300],
    #"n_estimators": [50, 100, 200, 300],
    "max_depth": [3],                 # ✅ keep shallow
   # "max_depth": [2, 3, 5],
    "learning_rate": [0.1],
    #"learning_rate": [0.01, 0.001, 0.0001],
    "subsample": [0.8],
    #"subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.9],
    #"colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1],
    #"min_child_weight": [1, 3, 5, 8],
    "gamma": [0],
    #"gamma": [0, 0.05, 0.1],
    #"reg_alpha": [0, 0.05, 0.1],
    "reg_alpha": [0],
    #"reg_lambda": [3, 5, 10],
    "reg_lambda": [5],
}
xgb_meta = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_meta, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
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
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

stacking_classifier = StackingClassifier(
    estimators=[('bayes', best_bayes), ('adaboost', best_adaboost), ('tree', best_tree), ('logistic', best_logistic)],
    final_estimator=best_xgb,
    passthrough=True,
    cv=5,
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

### Cell 43 — code (execution_count=None)
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

### Cell 44 — code (execution_count=None)
```python
# --------------------------------------------------
# XGBoost (Standalone) Performance
# --------------------------------------------------
print("\nEvaluating standalone XGBoost model...")

xgb_y_pred = best_xgb.predict(X_test)
xgb_y_pred_proba = best_xgb.predict_proba(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_precision = precision_score(y_test, xgb_y_pred, average='binary')
xgb_recall = recall_score(y_test, xgb_y_pred, average='binary')
xgb_f1 = f1_score(y_test, xgb_y_pred, average='binary')
xgb_roc_auc = roc_auc_score(y_test, xgb_y_pred_proba[:, 1])

print("\n" + "="*70)
print("XGBOOST (STANDALONE) PERFORMANCE METRICS")
print("="*70)
print(f"Accuracy:  {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
print(f"Precision: {xgb_precision:.4f} ({xgb_precision*100:.2f}%)")
print(f"Recall:    {xgb_recall:.4f} ({xgb_recall*100:.2f}%)")
print(f"F1-Score:  {xgb_f1:.4f} ({xgb_f1*100:.2f}%)")
print(f"ROC-AUC:   {xgb_roc_auc:.4f} ({xgb_roc_auc*100:.2f}%)")
print("="*70)

```

### Cell 45 — code (execution_count=None)
```python
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def collect_metrics(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall (Sensitivity)": recall_score(y_test, y_pred),
        "Specificity": tn / (tn + fp),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

results = []

results.append(
    collect_metrics(
        calibrated_stacking,
        X_test,
        y_test,
        "Calibrated Stacking (XGBoost Meta)"
    )
)

results.append(
    collect_metrics(
        best_xgb,
        X_test,
        y_test,
        "XGBoost (Standalone)"
    )
)

performance_table = pd.DataFrame(results)
performance_table = performance_table.set_index("Model").round(4)

print("\nMODEL PERFORMANCE COMPARISON")
print("="*80)
display(performance_table)

```

### Cell 46 — code (execution_count=None)
```python
def confusion_metrics_table(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return pd.DataFrame({
        "Metric": ["TP", "FP", "TN", "FN"],
        model_name: [tp, fp, tn, fn]
    })

cm_stack = confusion_metrics_table(
    calibrated_stacking, X_test, y_test, "Stacking"
)

cm_xgb = confusion_metrics_table(
    best_xgb, X_test, y_test, "XGBoost"
)

confusion_table = cm_stack.merge(cm_xgb, on="Metric")

print("\nCONFUSION MATRIX SUMMARY")
print("="*80)
display(confusion_table)

```

### Cell 47 — markdown (execution_count=None)
```text
## Mean ± SD (OSEC)
```

### Cell 48 — code (execution_count=None)
```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

ys66_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys66_cv = cross_validate(
    calibrated_stacking, X_train, y_train,
    cv=5, scoring=ys66_scoring
)

print("\n===== NOSEC (ys66) – Mean ± SD over 5-Fold CV =====")
for metric in ys66_scoring:
    scores = ys66_cv[f"test_{metric}"]
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")
print("=====================================================\n")

# ------------------------------------------------------------
# 7. Mean ± SD Plot
# ------------------------------------------------------------

ys66_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ys66_means = [
    ys66_cv['test_accuracy'].mean(),
    ys66_cv['test_precision'].mean(),
    ys66_cv['test_recall'].mean(),
    ys66_cv['test_f1'].mean(),
    ys66_cv['test_roc_auc'].mean()
]

ys66_stds = [
    ys66_cv['test_accuracy'].std(),
    ys66_cv['test_precision'].std(),
    ys66_cv['test_recall'].std(),
    ys66_cv['test_f1'].std(),
    ys66_cv['test_roc_auc'].std()
]

x = np.arange(len(ys66_metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys66_means,
    yerr=ys66_stds,
    fmt='o-', capsize=6, linewidth=2, markersize=8,
    color='darkred', ecolor='pink'
)

plt.xticks(x, ys66_metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Mean ± SD – NOSEC Stacking Classifier (ys66)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

```

### Cell 49 — markdown (execution_count=None)
```text

```

### Cell 50 — code (execution_count=None)
```python
# ✅ Step 1: Define base classifiers (from your variables)
base_classifiers = {
    "Naive Bayes": best_bayes,
    "AdaBoost": best_adaboost,
    "Decision Tree": best_tree,
    "Logistic Regression": best_logistic,
    "XGBoost": best_xgb,
    "Stacking (NOSEC)": calibrated_stacking
}

```

### Cell 51 — code (execution_count=None)
```python
# ✅ Step 2: Scoring dictionary
ys66_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

```

### Cell 52 — code (execution_count=None)
```python
#✅ Step 3: Cross-validate all base models + stacking
from sklearn.model_selection import cross_validate

cv_results = {}

for model_name, model in base_classifiers.items():
    print(f"Running 5-Fold CV for: {model_name}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=ys66_scoring,
        n_jobs=-1
    )

    cv_results[model_name] = scores

```

### Cell 53 — code (execution_count=None)
```python
# ✅ Step 4: Print Mean ± SD table (paper-ready)
print("\n===== Mean ± SD over 5-Fold CV =====")

for model_name, scores in cv_results.items():
    print(f"\n--- {model_name} ---")
    for metric in ys66_scoring:
        vals = scores[f"test_{metric}"]
        print(f"{metric.capitalize():<10}: {vals.mean():.4f} ± {vals.std():.4f}")

print("===================================")

```

### Cell 54 — code (execution_count=None)
```python
# ✅ Step 5: Mean ± SD Plot (same style as your original)
import numpy as np
import matplotlib.pyplot as plt

metrics_labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
x = np.arange(len(metrics_labels))

for model_name, scores in cv_results.items():

    means = [
        scores['test_accuracy'].mean(),
        scores['test_precision'].mean(),
        scores['test_recall'].mean(),
        scores['test_f1'].mean(),
        scores['test_roc_auc'].mean()
    ]

    stds = [
        scores['test_accuracy'].std(),
        scores['test_precision'].std(),
        scores['test_recall'].std(),
        scores['test_f1'].std(),
        scores['test_roc_auc'].std()
    ]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x, means,
        yerr=stds,
        fmt='o-', capsize=6, linewidth=2, markersize=8
    )

    plt.xticks(x, metrics_labels)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(f"Mean ± SD – {model_name}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

```

### Cell 55 — code (execution_count=None)
```python
# ✅ Combined Bar Chart (Mean ± SD for all models)
# 🔹 Step 1: Prepare mean & std arrays
import numpy as np
import matplotlib.pyplot as plt

models = list(cv_results.keys())
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

metric_labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

means = np.zeros((len(models), len(metrics)))
stds = np.zeros((len(models), len(metrics)))

for i, model in enumerate(models):
    for j, metric in enumerate(metrics):
        means[i, j] = cv_results[model][f"test_{metric}"].mean()
        stds[i, j] = cv_results[model][f"test_{metric}"].std()

# 🔹 Step 2: Plot combined bar chart
x = np.arange(len(models))
bar_width = 0.15

plt.figure(figsize=(14, 7))

for j, metric in enumerate(metric_labels):
    plt.bar(
        x + j * bar_width,
        means[:, j],
        bar_width,
        yerr=stds[:, j],
        capsize=5,
        label=metric
    )

plt.xticks(x + bar_width * 2, models, rotation=20)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Comparison of Base Classifiers and Stacking (Mean ± SD, 5-Fold CV)")
plt.legend(loc="upper left", ncol=3)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

```

### Cell 56 — code (execution_count=None)
```python
# 🔹 Explicit metric key mapping (best practice)
metric_keys = {
    "Accuracy": "test_accuracy",
    "Precision": "test_precision",
    "Recall": "test_recall",
    "F1-score": "test_f1",
    "ROC-AUC": "test_roc_auc"
}

import numpy as np
import matplotlib.pyplot as plt

models = list(cv_results.keys())
metrics = list(metric_keys.keys())

means = np.array([
    [cv_results[m][metric_keys[k]].mean() for k in metrics]
    for m in models
])

stds = np.array([
    [cv_results[m][metric_keys[k]].std() for k in metrics]
    for m in models
])

y = np.arange(len(models))
bar_height = 0.15
colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))

plt.figure(figsize=(12, 7))

for i, metric in enumerate(metrics):
    plt.barh(
        y + i * bar_height,
        means[:, i],
        xerr=stds[:, i],
        height=bar_height,
        color=colors[i],
        alpha=0.85,
        label=metric
    )

plt.yticks(y + bar_height * 2, models)
plt.xlabel("Score")
plt.xlim(0.90, 1.01)
plt.title(
    "Performance Comparison of Base Classifiers and NOSEC Stacking\n"
    "(Mean ± SD, 5-Fold CV)",
    fontsize=13
)

plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.legend(
    ncol=2,
    loc="lower right",
    frameon=True,
    fontsize=9
)

plt.tight_layout()
plt.show()

```

### Cell 57 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Metric key mapping (robust & explicit)
# ------------------------------------------------------------
metric_keys = {
    "Accuracy": "test_accuracy",
    "Precision": "test_precision",
    "Recall": "test_recall",
    "F1-score": "test_f1",
    "ROC-AUC": "test_roc_auc"
}

metrics = list(metric_keys.keys())
models = list(cv_results.keys())

# ------------------------------------------------------------
# 2. Extract mean and std values
# ------------------------------------------------------------
means = np.array([
    [cv_results[m][metric_keys[k]].mean() for k in metrics]
    for m in models
])

stds = np.array([
    [cv_results[m][metric_keys[k]].std() for k in metrics]
    for m in models
])

# ------------------------------------------------------------
# 3. Sort models by ROC-AUC (descending)
# ------------------------------------------------------------
roc_idx = metrics.index("ROC-AUC")
order = np.argsort(means[:, roc_idx])[::-1]

models = [models[i] for i in order]
means = means[order]
stds = stds[order]

# ------------------------------------------------------------
# 4. Plot settings
# ------------------------------------------------------------
y = np.arange(len(models))
bar_height = 0.13

colors = {
    "Accuracy": "#66c2a5",
    "Precision": "#8da0cb",
    "Recall": "#a6d854",
    "F1-score": "#e5c494",
    "ROC-AUC": "#b3b3b3"
}

plt.figure(figsize=(14, 8))

# ------------------------------------------------------------
# 5. Horizontal bar plot
# ------------------------------------------------------------
for i, metric in enumerate(metrics):
    plt.barh(
        y + i * bar_height,
        means[:, i],
        xerr=stds[:, i],
        height=bar_height,
        color=colors[metric],
        alpha=0.88,
        label=metric,
        error_kw=dict(alpha=0.6, lw=1.1, capsize=3)
    )

# ------------------------------------------------------------
# 6. Axis & labels
# ------------------------------------------------------------
plt.yticks(
    y + bar_height * 2,
    models,
    fontsize=11
)

plt.xlabel("Score", fontsize=12)
plt.xlim(0.88, 1.01)

plt.title(
    "Performance Comparison of Base Classifiers and NOSEC Stacking\n"
    "(Mean ± SD, 5-Fold Cross-Validation)",
    fontsize=14,
    pad=12
)

# ------------------------------------------------------------
# 8. Grid & legend
# ------------------------------------------------------------
plt.grid(axis="x", linestyle="--", alpha=0.4)

plt.legend(
    ncol=3,
    loc="lower left",
    fontsize=10,
    frameon=True
)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. Save for publication (recommended)
# ------------------------------------------------------------
plt.savefig("NOSEC_Model_Comparison.pdf", dpi=600, bbox_inches="tight")
plt.savefig("NOSEC_Model_Comparison.svg", bbox_inches="tight")

```

### Cell 58 — markdown (execution_count=None)
```text

```

### Cell 59 — code (execution_count=None)
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)


calibrated_stacking = CalibratedClassifierCV(
    estimator=stacking_classifier,
    method="sigmoid",   # Platt scaling
    cv=5
)

calibrated_stacking.fit(X_train, y_train)
y_pred_stack = calibrated_stacking.predict(X_test)
y_proba_stack = calibrated_stacking.predict_proba(X_test)[:, 1]
print("\n================= Calibrated Stacking Classifier =================")

print("Accuracy  :", accuracy_score(y_test, y_pred_stack))
print("Precision :", precision_score(y_test, y_pred_stack))
print("Recall    :", recall_score(y_test, y_pred_stack))
print("F1-score  :", f1_score(y_test, y_pred_stack))
print("ROC-AUC   :", roc_auc_score(y_test, y_proba_stack))
print("BrierScore:", brier_score_loss(y_test, y_proba_stack))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_stack))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_stack))

```

### Cell 60 — markdown (execution_count=None)
```text

```

### Cell 61 — markdown (execution_count=None)
```text
## 95% Confidence Intervals for AUC/F1
OSEC
```

### Cell 62 — code (execution_count=None)
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

### Cell 63 — code (execution_count=None)
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

### Cell 64 — code (execution_count=None)
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

### Cell 65 — markdown (execution_count=None)
```text
## Paired Bootstrap Significance Test Stacking (ALL metrics)
```

### Cell 66 — code (execution_count=None)
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

### Cell 67 — code (execution_count=None)
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

### Cell 68 — code (execution_count=None)
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

### Cell 69 — code (execution_count=None)
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

### Cell 70 — markdown (execution_count=None)
```text
## **Wilcoxon (Stacking vs Base models)**
```

### Cell 71 — code (execution_count=None)
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

### Cell 72 — code (execution_count=None)
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

### Cell 73 — code (execution_count=None)
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

### Cell 74 — markdown (execution_count=None)
```text
## Calibration (slope, intercept, Brier) and PR-AUC :::: Reviewer#2, Concern # 13:
```

### Cell 75 — code (execution_count=None)
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
    "n_estimators": [300],
    #"n_estimators": [50, 100, 200, 300],
    "max_depth": [3],                 # ✅ keep shallow
   # "max_depth": [2, 3, 5],
    "learning_rate": [0.1],
    #"learning_rate": [0.01, 0.001, 0.0001],
    "subsample": [0.8],
    #"subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.9],
    #"colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1],
    #"min_child_weight": [1, 3, 5, 8],
    "gamma": [0],
    #"gamma": [0, 0.05, 0.1],
    #"reg_alpha": [0, 0.05, 0.1],
    "reg_alpha": [0],
    #"reg_lambda": [3, 5, 10],
    "reg_lambda": [5],
}
xgb_meta = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_meta, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
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
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

stacking_classifier = StackingClassifier(
    estimators=[('bayes', best_bayes), ('adaboost', best_adaboost), ('tree', best_tree), ('logistic', best_logistic)],
    final_estimator=best_xgb,
    passthrough=True,
    cv=5,
    n_jobs=-1
)

# Calibrate the Stacking Classifier
calibrated_stacking = CalibratedClassifierCV(stacking_classifier, method='sigmoid', cv=cv, n_jobs=-1)

# Train the calibrated stacking classifier
print("\nTraining the Stacking Classifier with optimized hyperparameters...")
training_start = time.time()
calibrated_stacking.fit(X_train, y_train)
training_time = time.time() - training_start
print(f"Training completed in {training_time:.2f} seconds!")
```

### Cell 76 — code (execution_count=None)
```python
# ============================================================
# Calibration Metrics & PR-AUC for NOVEC Voting Classifier
# Reviewer #2 – Concern #13
# ============================================================

import numpy as np
from sklearn.metrics import (
    brier_score_loss, precision_recall_curve,
    auc, average_precision_score
)
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve

# ------------------------------------------------------------
# 1. Get predicted probabilities from NOVEC (positive class)
# ------------------------------------------------------------
NOVEC_prob = calibrated_stacking.predict_proba(X_test)[:, 1]

# True labels
y_true = y_test

# ------------------------------------------------------------
# 2. Brier Score (Calibration Quality)
# ------------------------------------------------------------
brier = brier_score_loss(y_true, NOVEC_prob)
print(f"Brier Score (NOVEC): {brier:.4f}")

# ------------------------------------------------------------
# 3. Calibration Slope & Calibration Intercept
#    Fit linear regression of:
#       predicted_probabilities (x) → actual outcomes (y)
# ------------------------------------------------------------

# Reshape probabilities for regression
NOVEC_prob_reshaped = NOVEC_prob.reshape(-1, 1)

# Linear regression: y_true = intercept + slope * predicted_prob
lr = LinearRegression().fit(NOVEC_prob_reshaped, y_true)

calib_intercept = lr.intercept_
calib_slope = lr.coef_[0]

print(f"Calibration Intercept (NOVEC): {calib_intercept:.4f}")
print(f"Calibration Slope (NOVEC):     {calib_slope:.4f}")

# ------------------------------------------------------------
# 4. PR-AUC (Precision–Recall AUC)
# ------------------------------------------------------------

precision, recall, _ = precision_recall_curve(y_true, NOVEC_prob)

# PR-AUC using trapezoidal integration
pr_auc = auc(recall, precision)

# AP score (area under PR curve using average precision)
ap_score = average_precision_score(y_true, NOVEC_prob)

print(f"PR-AUC (NOVEC):                 {pr_auc:.4f}")
print(f"Average Precision Score (NOVEC): {ap_score:.4f}")

# ------------------------------------------------------------
# Optional: Plot Calibration Curve
# ------------------------------------------------------------

import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true, NOVEC_prob, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker='o', label='NOVEC Calibration')
plt.plot([0, 1], [0, 1], linestyle='--', color='orange', label='Perfect Calibration')

plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve – NOVEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.show()

```

### Cell 77 — code (execution_count=None)
```python
# ----------------------------------------------------------
# 4. Predictions & Core Metrics
# ----------------------------------------------------------

stacking_prob_platt = calibrated_stacking.predict_proba(X_test)[:, 1]
stacking_pred_platt = (stacking_prob_platt >= 0.5).astype(int)

train_accuracy = calibrated_stacking.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, stacking_pred_platt)
precision = precision_score(y_test, stacking_pred_platt)
recall = recall_score(y_test, stacking_pred_platt)
f1 = f1_score(y_test, stacking_pred_platt)

# ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, stacking_prob_platt)
auc_score = auc(fpr, tpr)

# PR-AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, stacking_prob_platt)
pr_auc = auc(recall_curve, precision_curve)

# Average Precision Score
avg_precision = average_precision_score(y_test, stacking_prob_platt)

# Brier Score
brier = brier_score_loss(y_test, stacking_prob_platt)


# ----------------------------------------------------------
# 4A. Calibration Slope & Intercept
# ----------------------------------------------------------

probs_reshaped = stacking_prob_platt.reshape(-1, 1)
calib_lr = LogisticRegression().fit(probs_reshaped, y_test)

calib_slope = calib_lr.coef_[0][0]
calib_intercept = calib_lr.intercept_[0]


# ----------------------------------------------------------
# 5. Print All Metrics
# ----------------------------------------------------------

print("========== Platt-Calibrated OSEC Model ==========")
print(f"Training Accuracy       : {train_accuracy:.4f}")
print(f"Testing Accuracy        : {test_accuracy:.4f}")
print(f"Precision               : {precision:.4f}")
print(f"Recall                  : {recall:.4f}")
print(f"F1 Score                : {f1:.4f}")
print(f"AUC-ROC                 : {auc_score:.4f}")
print(f"PR-AUC                  : {pr_auc:.4f}")
print(f"Average Precision Score : {avg_precision:.4f}")
print(f"Brier Score             : {brier:.4f}")
print(f"Calibration Intercept   : {calib_intercept:.4f}")
print(f"Calibration Slope       : {calib_slope:.4f}")
print("===================================================")


# ----------------------------------------------------------
# 6. Plot ROC Curve
# ----------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Calibrated OSEC AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.title('ROC Curve – Platt-Calibrated OSEC Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 7. Plot Calibration Curve
# ----------------------------------------------------------

prob_true, prob_pred = calibration_curve(y_test, stacking_prob_platt, n_bins=10)

plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', label="Calibrated Stacking Classifier")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve (Reliability Plot)")
plt.legend()
plt.grid(True)
plt.show()


# ----------------------------------------------------------
# 8. Plot Precision–Recall (PR) Curve
# ----------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")

# Baseline = positive class ratio
baseline = np.mean(y_test)
plt.hlines(baseline, xmin=0, xmax=1, color='orange', linestyles='--', label=f'Baseline = {baseline:.3f}')

plt.title("Precision–Recall Curve – Platt-Calibrated OSEC Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()

```

### Cell 78 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# ----------------------------------------------------------
# 7. Plot Calibration Curve (Save as PDF 300 DPI)
# ----------------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated Proposed OSEC')
plt.plot([0, 1], [0, 1], '--', color='orange', label='Perfect Calibration')
#plt.title("Calibration Curve – Platt Scaling")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.legend(loc="upper left")
plt.grid(True)

calibration_path = "/content/calibration_curve_300dpi.pdf"
plt.savefig(calibration_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# Auto-download the PDF
files.download(calibration_path)


# ----------------------------------------------------------
# 8. Plot Precision–Recall (PR) Curve (Save as PDF 300 DPI)
# ----------------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, linewidth=2,
         label=f"PR-AUC of Platt-Calibrated Proposed OSEC  = {pr_auc:.4f}")

baseline = np.mean(y_test)
plt.hlines(baseline, xmin=0, xmax=1, color='orange', linestyles='--',
           label=f'Baseline (Logistic Regression) = {baseline:.4f}')

#plt.title("Precision–Recall Curve – Platt-Calibrated NOVEC Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()

pr_curve_path = "/content/pr_curve_300dpi.pdf"
plt.savefig(pr_curve_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# Auto-download the PDF
files.download(pr_curve_path)

```

### Cell 79 — code (execution_count=None)
```python
y_prob = NOVEC_prob
ece, info = expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10,
    strategy="uniform"   # or "quantile"
)

print(f"{'Bin':<3} {'Range':<15} {'Count':<7} {'Frac':<8} "
      f"{'Conf':<8} {'Acc':<8} {'|Acc-Conf|':<12} {'Contribution'}")

for i in range(len(info["bin_counts"])):
    count = info["bin_counts"][i]
    frac = info["bin_fraction"][i]
    conf = info["bin_confidence"][i]
    acc = info["bin_accuracy"][i]

    if count == 0:
        print(f"{i:<3} EMPTY")
        continue

    contrib = frac * abs(acc - conf)
    bin_range = f"[{info['bin_edges'][i]:.2f}, {info['bin_edges'][i+1]:.2f}]"

    print(f"{i:<3} {bin_range:<15} {count:<7} {frac:<8.3f} "
          f"{conf:<8.3f} {acc:<8.3f} {abs(acc-conf):<12.3f} {contrib:.5f}")

print("\nTotal ECE:", ece)

```

### Cell 80 — code (execution_count=None)
```python
def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, qs)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    ece = 0.0
    n = len(y_true)

    bin_acc = []
    bin_conf = []
    bin_frac = []
    bin_counts = []

    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        count = mask.sum()

        if count == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_frac.append(0.0)
            bin_counts.append(0)
            continue

        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        frac = count / n

        ece += frac * abs(acc - conf)

        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_frac.append(frac)
        bin_counts.append(count)

    bin_info = {
        "bin_edges": bin_edges,
        "bin_accuracy": np.array(bin_acc),
        "bin_confidence": np.array(bin_conf),
        "bin_fraction": np.array(bin_frac),
        "bin_counts": np.array(bin_counts),
    }

    return ece, bin_info
```

### Cell 81 — code (execution_count=None)
```python
ece, info = expected_calibration_error(
    y_true,
    NOVEC_prob,
    n_bins=10,
    strategy="uniform"
)

print(f"{'Bin':<3} {'Range':<15} {'Count':<7} {'Frac':<8} "
      f"{'Conf':<8} {'Acc':<8} {'|Acc-Conf|':<12} {'Contribution'}")

for i in range(len(info["bin_counts"])):
    count = info["bin_counts"][i]
    frac = info["bin_fraction"][i]
    conf = info["bin_confidence"][i]
    acc = info["bin_accuracy"][i]

    if count == 0:
        print(f"{i:<3} EMPTY")
        continue

    contrib = frac * abs(acc - conf)
    bin_range = f"[{info['bin_edges'][i]:.2f}, {info['bin_edges'][i+1]:.2f}]"

    print(f"{i:<3} {bin_range:<15} {count:<7} {frac:<8.3f} "
          f"{conf:<8.3f} {acc:<8.3f} {abs(acc-conf):<12.3f} {contrib:.5f}")

print("\nTotal ECE:", ece)

```

### Cell 82 — code (execution_count=None)
```python
print(type(ece), ece)
print(info.keys())

```

### Cell 83 — code (execution_count=None)
```python
import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, qs)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
        bin_edges = np.unique(bin_edges)  # avoid duplicate edges
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    ece = 0.0
    n = len(y_true)

    bin_acc, bin_conf, bin_frac, bin_counts = [], [], [], []

    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        count = mask.sum()

        if count == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_frac.append(0.0)
            bin_counts.append(0)
            continue

        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        frac = count / n

        ece += frac * abs(acc - conf)

        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_frac.append(frac)
        bin_counts.append(count)

    info = {
        "bin_edges": bin_edges,
        "bin_accuracy": np.array(bin_acc),
        "bin_confidence": np.array(bin_conf),
        "bin_fraction": np.array(bin_frac),
        "bin_counts": np.array(bin_counts),
    }

    return ece, info

```

### Cell 84 — code (execution_count=None)
```python
def print_bin_table(info, ece, title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

    print(f"{'Bin':<3} {'Range':<15} {'Count':<7} {'Frac':<8} "
          f"{'Conf':<8} {'Acc':<8} {'|Acc-Conf|':<12} {'Contribution'}")

    for i in range(len(info["bin_counts"])):
        count = info["bin_counts"][i]
        if count == 0:
            print(f"{i:<3} EMPTY")
            continue

        frac = info["bin_fraction"][i]
        conf = info["bin_confidence"][i]
        acc = info["bin_accuracy"][i]
        contrib = frac * abs(acc - conf)
        bin_range = f"[{info['bin_edges'][i]:.2f}, {info['bin_edges'][i+1]:.2f}]"

        print(f"{i:<3} {bin_range:<15} {count:<7} {frac:<8.3f} "
              f"{conf:<8.3f} {acc:<8.3f} {abs(acc-conf):<12.3f} {contrib:.5f}")

    print("\nECE:", ece)

```

### Cell 85 — code (execution_count=None)
```python
y_prob = NOVEC_prob   # your probabilities
# y_true = ...        # your true labels

# --- Uniform bins (diagnostics) ---
ece_u, info_u = expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10,
    strategy="uniform"
)

print_bin_table(
    info_u,
    ece_u,
    title="ECE with Uniform Bins (10) — Diagnostic"
)

# --- Quantile bins (reporting) ---
ece_q, info_q = expected_calibration_error(
    y_true,
    y_prob,
    n_bins=5,
    strategy="quantile"
)

print_bin_table(
    info_q,
    ece_q,
    title="ECE with Quantile Bins (5) — Reporting"
)

```

### Cell 86 — markdown (execution_count=None)
```text
Calibration Curve (Uniform Bins — Diagnostic)
```

### Cell 87 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()  # prevent duplicate rendering in notebooks

```

### Cell 88 — code (execution_count=None)
```python
def compute_uniform_bin_stats(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    mean_pred, frac_pos, bin_counts, bin_ranges = [], [], [], []

    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue

        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
        bin_counts.append(count)
        bin_ranges.append((bin_edges[b], bin_edges[b + 1]))

    return (
        np.array(mean_pred),
        np.array(frac_pos),
        np.array(bin_counts),
        bin_ranges
    )

```

### Cell 89 — code (execution_count=None)
```python
def smart_annotate_no_overlap(ax, x, y, text, placed, y_step=0.06):
    """
    Collision-aware annotation placement.
    Ensures labels never overlap.
    """

    # Prefer below for high y, above for low y
    dy = -0.06 if y > 0.75 else 0.06
    new_y = y + dy

    # Shift label until no collision
    while any(abs(new_y - py) < y_step and abs(x - px) < 0.12
              for px, py in placed):
        new_y += y_step if dy > 0 else -y_step

    placed.append((x, new_y))

    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, new_y),
        textcoords="data",
        ha="center",
        va="bottom" if new_y > y else "top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.95),
        arrowprops=dict(arrowstyle="->", lw=0.85)
    )

```

### Cell 90 — code (execution_count=None)
```python
def plot_uniform_calibration_ece_aligned(
    y_true,
    y_prob,
    n_bins=10,
    filename="uniform_calibration_diagnostic_300dpi.pdf"
):
    mean_pred, frac_pos, bin_counts, bin_ranges = compute_uniform_bin_stats(
        y_true, y_prob, n_bins
    )

    fig, ax = plt.subplots(figsize=(11, 11))

    # Model curve
    ax.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2.5,
        label="Proposed OSEC Stacking Model"
    )

    # Perfect calibration
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Perfect calibration"
    )

    # Collision-aware annotations
    placed_labels = []
    for x, y, c, (l, r) in zip(mean_pred, frac_pos, bin_counts, bin_ranges):
        label = f"[{l:.1f}, {r:.1f}]\n n={c}\n({x:.2f}, {y:.2f})"
        smart_annotate_no_overlap(ax, x, y, label, placed_labels)

    ax.set_xlabel("Mean predicted probability", fontsize=14)
    ax.set_ylabel("Observed fraction of positives", fontsize=14)
    #ax.set_title("Calibration Curve (Uniform Bins — Diagnostic)", fontsize=13)
    ax.legend(frameon=False, fontsize=14)
    ax.grid(alpha=0.5)

    plt.tight_layout()

    # ✅ SAVE (vector PDF, 300 DPI)
    plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {filename}")

```

### Cell 91 — code (execution_count=None)
```python
plot_uniform_calibration_ece_aligned(
    y_true,
    NOVEC_prob,
    n_bins=10,
    filename="uniform_calibration_diagnostic_300dpi.pdf"
)

from google.colab import files
files.download("uniform_calibration_diagnostic_300dpi.pdf")

```

### Cell 92 — code (execution_count=None)
```python

```

### Cell 93 — code (execution_count=None)
```python
ece_u, info_u = expected_calibration_error(
    y_true,
    NOVEC_prob,
    n_bins=10,
    strategy="uniform"
)

print_bin_table(
    info_u,
    ece_u,
    title="ECE with Uniform Bins (10) — Diagnostic Only"
)

```

### Cell 94 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_quantile_calibration_publication(
    y_true,
    y_prob,
    n_bins=5,
    title="Calibration Curve (Quantile Bins — Reporting)"
):
    # Compute calibration curve with quantile bins
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile"
    )

    plt.figure(figsize=(4.5, 4.5))

    # Model curve
    plt.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        label="Proposed OSEC Stacking Model"
    )

    # Perfect calibration reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration"
    )

    # Styling for publication
    plt.xlabel("Mean predicted probability", fontsize=11)
    plt.ylabel("Observed fraction of positives", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(frameon=False, fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()

```

### Cell 95 — code (execution_count=None)
```python
plot_quantile_calibration_publication(
    y_true,
    NOVEC_prob,
    n_bins=5
)

```

### Cell 96 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_quantile_calibration_with_counts(
    y_true,
    y_prob,
    n_bins=5,
    title="Calibration Curve (Quantile Bins — Reporting)"
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Calibration curve (quantile bins)
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile"
    )

    # Compute bin counts manually (same quantile edges)
    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    bin_counts = np.array([(bin_ids == i).sum() for i in range(n_bins)])

    # Plot
    plt.figure(figsize=(4.5, 4.5))

    plt.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        label="Model"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration"
    )

    # Annotate bin counts
    for x, y, c in zip(mean_pred, frac_pos, bin_counts):
        plt.annotate(
            f"n={c}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9
        )

    # Styling
    plt.xlabel("Mean predicted probability", fontsize=11)
    plt.ylabel("Observed fraction of positives", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(frameon=False, fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()

```

### Cell 97 — code (execution_count=None)
```python
plot_quantile_calibration_with_counts(
    y_true,
    NOVEC_prob,
    n_bins=5
)

```

### Cell 98 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def save_quantile_calibration_pdf(
    y_true,
    y_prob,
    n_bins=5,
    filename="calibration_quantile_300dpi.pdf",
    #title="Calibration Curve (Quantile Bins — Reporting)"
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Calibration curve (quantile bins)
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile"
    )

    # Compute quantile bin counts
    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    bin_counts = np.array([(bin_ids == i).sum() for i in range(n_bins)])

    # Plot
    plt.figure(figsize=(4.5, 4.5))

    plt.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        label="Proposed OSEC Stacking Model"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration"
    )

    # Annotate bin counts
    for x, y, c in zip(mean_pred, frac_pos, bin_counts):
        plt.annotate(
            f"n={c}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9
        )

    # Styling
    plt.xlabel("Mean predicted probability", fontsize=11)
    plt.ylabel("Observed fraction of positives", fontsize=11)
    #plt.title(title, fontsize=12)
    plt.legend(frameon=False, fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save as 300-DPI PDF
    plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved publication-ready figure to: {filename}")

```

### Cell 99 — code (execution_count=None)
```python
save_quantile_calibration_pdf(
    y_true,
    NOVEC_prob,
    n_bins=5,
    filename="calibration_quantile_300dpi.pdf"
)

from google.colab import files
files.download("calibration_quantile_300dpi.pdf")

```

### Cell 100 — markdown (execution_count=None)
```text
## Decision Curve Analysis (DCA)
```

### Cell 101 — code (execution_count=None)
```python
# ============================================================
# Decision Curve Analysis (DCA) – Final Revised Code
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. DCA Function (Corrected + Clean)
# ------------------------------------------------------------
def decision_curve_analysis(y_true, y_prob,
                            thresholds=np.linspace(0.01, 0.99, 100)):
    """
    Compute net benefit across threshold probabilities.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    thresholds : array-like
        Threshold probability values for DCA.

    Returns:
    --------
    thresholds : array-like
        Thresholds used in DCA.
    net_benefit : numpy array
        Net benefit values corresponding to thresholds.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    net_benefit = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        # Standard net benefit formula:
        # NB = (TP/n) - (FP/n) * (t / (1 - t))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(nb)

    return thresholds, np.array(net_benefit)


# ------------------------------------------------------------
# 2. Inputs for NOVEC Stacking Classifier
# ------------------------------------------------------------
# Replace with your actual variables:
# y_test_old : true labels
# novec_prob : predicted probabilities of NOVEC classifier

y_true = y_test

Stacking_prob_platt = calibrated_stacking.predict_proba(X_test)[:, 1]
Stacking_pred_platt = (Stacking_prob_platt >= 0.5).astype(int)


y_prob = Stacking_prob_platt   # shape (n_samples,)


# ------------------------------------------------------------
# 3. Compute DCA Values
# ------------------------------------------------------------
thresholds, net_benefit_novec = decision_curve_analysis(y_true, y_prob)

# Treat None strategy (always zero)
nb_treat_none = np.zeros_like(thresholds)

# Treat All strategy (depends on prevalence)
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


# ------------------------------------------------------------
# 4. Plot DCA Curve
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(thresholds, net_benefit_novec, label="OSEC Stacking Classifier", linewidth=2.1)
plt.plot(thresholds, nb_treat_none, '--', label="Treat None", linewidth=1.8)
plt.plot(thresholds, nb_treat_all, '--', label="Treat All", linewidth=1.8)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA) – OSEC Stacking Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

### Cell 102 — code (execution_count=None)
```python
from google.colab import files

# ------------------------------------------------------------
# 4. Plot & Save DCA Curve (300 DPI PDF + Auto-Download)
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(thresholds, net_benefit_novec, label="Proposed OSEC Stacking Ensemble Classifier", linewidth=2.1)
plt.plot(thresholds, nb_treat_none, '--', label="Treat None", linewidth=1.8)
plt.plot(thresholds, nb_treat_all, '--', label="Treat All", linewidth=1.8)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
#plt.title("Decision Curve Analysis (DCA) – OSEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save PDF at 300 dpi
dca_pdf_path = "/content/DCA_OVEC_300dpi.pdf"
plt.savefig(dca_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# Auto-download PDF
files.download(dca_pdf_path)

```

### Cell 103 — code (execution_count=None)
```python
# ------------------------------------------------------------
# 3. Compute DCA Values
# ------------------------------------------------------------
thresholds, net_benefit_novec = decision_curve_analysis(y_true, y_prob)

# ------------------------------------------------------------
# 3A. Print Net Benefit Along With Threshold
# ------------------------------------------------------------
print("\n=== Threshold vs Net Benefit (OSEC Stacking Classifier) ===")
print(f"{'Threshold':<12} {'Net Benefit':<12}")

for t, nb in zip(thresholds, net_benefit_novec):
    print(f"{t:<12.4f} {nb:<12.6f}")

# Treat None strategy (always zero)
nb_treat_none = np.zeros_like(thresholds)

# Treat All strategy (depends on prevalence)
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

```

### Cell 104 — markdown (execution_count=None)
```text
## Proposed OVEC
```

### Cell 105 — code (execution_count=None)
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
    'solver': ['liblinear', 'lbfgs']
}
log_rs = RandomizedSearchCV(LogisticRegression(), param_log, n_iter=10, cv=5, scoring='accuracy', random_state=42)
log_rs.fit(X_train, y_train)
logistic_best = log_rs.best_estimator_
print(f"✓ Best Logistic Regression params: {log_rs.best_estimator_}")
print(f"✓ Best Logistic Regression CV score: {log_rs.best_score_:.4f}\n")

# 2. Random Forest
param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [2, None],
    'criterion': ['gini', 'entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train, y_train)
rf_best = rf_rs.best_estimator_

print(f"✓ Best Random Forest params: {rf_rs.best_estimator_}")
print(f"✓ Best Random Forest CV score: {rf_rs.best_score_:.4f}\n")


# 3. KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train, y_train)
knn_best = knn_rs.best_estimator_

print(f"✓ Best KNN params: {knn_rs.best_estimator_}")
print(f"✓ Best KNN CV score: {knn_rs.best_score_:.4f}\n")

# 4. SVM
param_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train, y_train)
svm_best = svm_rs.best_estimator_

print(f"✓ Best svm params: {svm_rs.best_estimator_}")
print(f"✓ Best svm CV score: {svm_rs.best_score_:.4f}\n")


# Combine tuned models into VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ('knn', knn_best),
        ('rf', rf_best),
        ('logistic', logistic_best),
        ('svm', svm_best)
    ],
    voting='soft',
    weights=[1, 2, 1, 2]  # Example weights
)

# Fit the model
voting_model.fit(X_train, y_train)

# Make predictions
voting_predictions = voting_model.predict(X_test)

# Evaluate Metrics
train_accuracy = voting_model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, voting_predictions)
precision = precision_score(y_test, voting_predictions)
recall = recall_score(y_test, voting_predictions)
f1 = f1_score(y_test, voting_predictions)

# AUC-ROC Curve
voting_probabilities = voting_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities)
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
print(f"Testing Accuracy : {test_accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"AUC-ROC           : {auc_score:.4f}")
print("===========================================================")

```

### Cell 106 — code (execution_count=None)
```python
print(f"✓ Best Logistic Regression params: {log_rs.best_estimator_}")
print(f"✓ Best Logistic Regression CV score: {log_rs.best_score_:.4f}\n")

print(f"✓ Best Random Forest params: {rf_rs.best_estimator_}")
print(f"✓ Best Random Forest CV score: {rf_rs.best_score_:.4f}\n")

print(f"✓ Best KNN params: {knn_rs.best_estimator_}")
print(f"✓ Best KNN CV score: {knn_rs.best_score_:.4f}\n")

print(f"✓ Best svm params: {svm_rs.best_estimator_}")
print(f"✓ Best svm CV score: {svm_rs.best_score_:.4f}\n")

```

### Cell 107 — code (execution_count=None)
```python
# Evaluate Metrics
train_accuracy_vot = voting_model.score(X_train, y_train)
test_accuracy_vot = accuracy_score(y_test, voting_predictions)
precision_vot = precision_score(y_test, voting_predictions)
recall_vot = recall_score(y_test, voting_predictions)
f1_vot = f1_score(y_test, voting_predictions)

# AUC-ROC Curve
voting_probabilities_vot = voting_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities_vot)
auc_score_vot = auc(fpr, tpr)

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
print(f"Training Accuracy: {train_accuracy_vot:.4f}")
print(f"Testing Accuracy : {test_accuracy_vot:.4f}")
print(f"Precision         : {precision_vot:.4f}")
print(f"Recall            : {recall_vot:.4f}")
print(f"F1 Score          : {f1_vot:.4f}")
print(f"AUC-ROC           : {auc_score_vot:.4f}")
print("===========================================================")
```

### Cell 108 — code (execution_count=None)
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
voting_predictions_vot = voting_model.predict(X_test)
cm_vot = confusion_matrix(y_test, voting_predictions_vot)
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_vot)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Voting Classifier")
plt.grid(False)
plt.tight_layout()
plt.show()

```

### Cell 109 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities (for AUC)
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_test_prob)
    else:
        auc = "N/A"

    print(f"\n========== {name} Performance ==========")
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Testing Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precision        : {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall           : {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1 Score         : {f1_score(y_test, y_test_pred):.4f}")
    print(f"AUC-ROC          : {auc}")
    print("========================================")

evaluate_model("Logistic Regression", logistic_best, X_train, X_test, y_train, y_test)
evaluate_model("Random Forest", rf_best, X_train, X_test, y_train, y_test)
evaluate_model("KNN", knn_best, X_train, X_test, y_train, y_test)
evaluate_model("SVM", svm_best, X_train, X_test, y_train, y_test)
# Ensemble model
evaluate_model("Weighted Voting Classifier", voting_model, X_train, X_test, y_train, y_test)
```

### Cell 110 — markdown (execution_count=None)
```text
## 95% Confidence Intervals for AUC/F1
OVEC
```

### Cell 111 — code (execution_count=None)
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

### Cell 112 — code (execution_count=None)
```python
# Fit voting classifier on training data
voting_model.fit(X_train, y_train)


# 1️⃣ Store models (already tuned in your pipeline)
models = {
    "KNN": knn_best,
    "SVM": svm_best,
    "Logistic Regression": logistic_best,
    "Random Forest": rf_best,
    "Voting": voting_model
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

### Cell 113 — code (execution_count=None)
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

### Cell 114 — code (execution_count=None)
```python

```

### Cell 115 — markdown (execution_count=None)
```text
## Paired Bootstrap Significance Test Stacking (ALL metrics)
```

### Cell 116 — code (execution_count=None)
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

### Cell 117 — code (execution_count=None)
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

### Cell 118 — code (execution_count=None)
```python
# Apply (Stacking vs Base models)
base_models = {
    "KNN": knn_best,
    "SVM": svm_best,
    "Logistic Regression": logistic_best,
    "Random Forest": rf_best,
}

bootstrap_tests = {}

print("\nPaired Bootstrap Tests (Voting vs Base Models)\n")

for name, model in base_models.items():
    print(f"Bootstrap: Voting vs {name}")

    bootstrap_tests[name] = paired_bootstrap_all_metrics(
        calibrated_stacking,
        model,
        X_test,
        y_test,
        metric_fns,
        n_bootstraps=1000
    )

```

### Cell 119 — code (execution_count=None)
```python
# Bootstrap results
print("\n" + "="*100)
print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS (n = 1000)")
print("="*100)

for model, metrics in bootstrap_tests.items():
    print(f"\nVoting vs {model}")
    print("-"*100)
    for m, res in metrics.items():
        print(
            f"{m.upper():9s} | Δ = {res['mean_diff']:+.4f} | p = {res['p_value']:.6f}"
        )

```

### Cell 120 — markdown (execution_count=None)
```text
## **Wilcoxon (Voting vs Base models)**
```

### Cell 121 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, y_loras, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 122 — code (execution_count=None)
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

### Cell 123 — code (execution_count=None)
```python
# Wilcoxon results
wilcoxon_tests = {}

print("\nWilcoxon Tests Across CV Folds\n")

for name, model in base_models.items():
    print(f"Wilcoxon: Voting vs {name}")

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

### Cell 124 — code (execution_count=None)
```python
print("\n" + "="*100)
print("WILCOXON SIGNED-RANK TESTS (5-Fold CV)")
print("="*100)

for model, metrics in wilcoxon_tests.items():
    print(f"\nVoting vs {model}")
    print("-"*100)
    for m, res in metrics.items():
        print(
            f"{m.upper():9s} | stat = {res['stat']:.4f} | p = {res['p_value']:.6f}"
        )
```

### Cell 125 — markdown (execution_count=None)
```text

```

### Cell 126 — code (execution_count=None)
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
    'solver': ['liblinear', 'lbfgs']
}
log_rs = RandomizedSearchCV(LogisticRegression(), param_log, n_iter=10, cv=5, scoring='accuracy', random_state=42)
log_rs.fit(X_train, y_train)
logistic_best = log_rs.best_estimator_
print(f"✓ Best Logistic Regression params: {log_rs.best_estimator_}")
print(f"✓ Best Logistic Regression CV score: {log_rs.best_score_:.4f}\n")

# 2. Random Forest
param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [2, None],
    'criterion': ['gini', 'entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train, y_train)
rf_best = rf_rs.best_estimator_

print(f"✓ Best Random Forest params: {rf_rs.best_estimator_}")
print(f"✓ Best Random Forest CV score: {rf_rs.best_score_:.4f}\n")


# 3. KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train, y_train)
knn_best = knn_rs.best_estimator_

print(f"✓ Best KNN params: {knn_rs.best_estimator_}")
print(f"✓ Best KNN CV score: {knn_rs.best_score_:.4f}\n")

# 4. SVM
param_svm = {
    'C': [100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train, y_train)
svm_best = svm_rs.best_estimator_

print(f"✓ Best svm params: {svm_rs.best_estimator_}")
print(f"✓ Best svm CV score: {svm_rs.best_score_:.4f}\n")
```

### Cell 127 — code (execution_count=None)
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

platt_calibrated_model = CalibratedClassifierCV(
    estimator=voting_model,
    method='sigmoid',
    cv=cv,
    n_jobs=-1
)

platt_calibrated_model.fit(X_train, y_train)
```

### Cell 128 — code (execution_count=None)
```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon

# 1️⃣ Define repeated stratified CV splits (SHARED BY ALL MODELS)
def get_repeated_cv_splits(X, y, n_splits=10, n_runs=2):
    splits = []
    for seed in range(n_runs):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        for tr_idx, val_idx in skf.split(X, y):
            splits.append((tr_idx, val_idx))
    return splits

```

### Cell 129 — code (execution_count=None)
```python
# 2️⃣ Evaluate a model on the shared splits
def evaluate_model_on_splits(model, X, y, splits, metric_fns):
    scores = {m: [] for m in metric_fns}

    for tr_idx, val_idx in splits:
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        for name, fn in metric_fns.items():
            scores[name].append(fn(y_val, y_pred, y_prob))

    return scores

```

### Cell 130 — code (execution_count=None)
```python
# 3️⃣ Generate shared CV splits (ONCE)
splits = get_repeated_cv_splits(
    X_train, y_train,
    n_splits=10,
    n_runs=2
)

```

### Cell 131 — code (execution_count=None)
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

# 4️⃣ Evaluate Voting / Stacking model
voting_scores = evaluate_model_on_splits(
    platt_calibrated_model,
    X_train,
    y_train,
    splits,
    metric_fns
)

```

### Cell 132 — code (execution_count=None)
```python

# 5️⃣ Evaluate all base models on the SAME splits
base_scores = {}

base_models = {
    "KNN": knn_best,
    "SVM": svm_best,
    "Logistic Regression": logistic_best,
    "Random Forest": rf_best,
}

for name, model in base_models.items():
    base_scores[name] = evaluate_model_on_splits(
        model,
        X_train,
        y_train,
        splits,
        metric_fns
    )

```

### Cell 133 — code (execution_count=None)
```python
# 6️⃣ Apply Wilcoxon Signed-Rank Test (CORRECTLY)
wilcoxon_tests = {}

for model_name in base_models:
    wilcoxon_tests[model_name] = {}

    for metric in metric_fns:
        stat, p = wilcoxon(
            voting_scores[metric],
            base_scores[model_name][metric],
            alternative="greater"   # Voting > Baseline
        )

        wilcoxon_tests[model_name][metric] = {
            "stat": stat,
            "p_value": p
        }

```

### Cell 134 — code (execution_count=None)
```python
# 7️⃣ Print results (YOUR PRINT CODE — SLIGHTLY UPDATED HEADER)
print("\n" + "="*100)
print("WILCOXON SIGNED-RANK TESTS (10×10 Repeated Stratified CV)")
print("="*100)

for model, metrics in wilcoxon_tests.items():
    print(f"\nVoting vs {model}")
    print("-"*100)
    for m, res in metrics.items():
        print(
            f"{m.upper():9s} | stat = {res['stat']:.4f} | p = {res['p_value']:.6f}"
        )

```

### Cell 135 — markdown (execution_count=None)
```text
## Calibration (slope, intercept, Brier) and PR-AUC :::: Reviewer#2, Concern # 13:
```

### Cell 136 — code (execution_count=None)
```python
# ============================================================
# Calibration Metrics & PR-AUC for NOVEC Voting Classifier
# Reviewer #2 – Concern #13
# ============================================================

import numpy as np
from sklearn.metrics import (
    brier_score_loss, precision_recall_curve,
    auc, average_precision_score
)
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve

# ------------------------------------------------------------
# 1. Get predicted probabilities from NOVEC (positive class)
# ------------------------------------------------------------
NOVEC_prob = voting_model.predict_proba(X_test)[:, 1]

# True labels
y_true = y_test

# ------------------------------------------------------------
# 2. Brier Score (Calibration Quality)
# ------------------------------------------------------------
brier = brier_score_loss(y_true, NOVEC_prob)
print(f"Brier Score (NOVEC): {brier:.4f}")

# ------------------------------------------------------------
# 3. Calibration Slope & Calibration Intercept
#    Fit linear regression of:
#       predicted_probabilities (x) → actual outcomes (y)
# ------------------------------------------------------------

# Reshape probabilities for regression
NOVEC_prob_reshaped = NOVEC_prob.reshape(-1, 1)

# Linear regression: y_true = intercept + slope * predicted_prob
lr = LinearRegression().fit(NOVEC_prob_reshaped, y_true)

calib_intercept = lr.intercept_
calib_slope = lr.coef_[0]

print(f"Calibration Intercept (OVEC): {calib_intercept:.4f}")
print(f"Calibration Slope (OVEC):     {calib_slope:.4f}")

# ------------------------------------------------------------
# 4. PR-AUC (Precision–Recall AUC)
# ------------------------------------------------------------

precision, recall, _ = precision_recall_curve(y_true, NOVEC_prob)

# PR-AUC using trapezoidal integration
pr_auc = auc(recall, precision)

# AP score (area under PR curve using average precision)
ap_score = average_precision_score(y_true, NOVEC_prob)

print(f"PR-AUC (OVEC):                 {pr_auc:.4f}")
print(f"Average Precision Score (OVEC): {ap_score:.4f}")

# ------------------------------------------------------------
# Optional: Plot Calibration Curve
# ------------------------------------------------------------

import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true, NOVEC_prob, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker='o', label='OVEC Calibration')
plt.plot([0, 1], [0, 1], linestyle='--', color='orange', label='Perfect Calibration')

plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve – OVEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.show()

```

### Cell 137 — code (execution_count=None)
```python

```

### Cell 138 — code (execution_count=None)
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

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

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



# ----------------------------------------------------------
# 3. Apply Platt Scaling (Sigmoid Calibration)
# ----------------------------------------------------------

platt_calibrated_model = CalibratedClassifierCV(
    estimator=voting_model,
    method='sigmoid',
    cv=cv,
    n_jobs=-1
)

platt_calibrated_model.fit(X_train, y_train)

```

### Cell 139 — code (execution_count=None)
```python
# ----------------------------------------------------------
# 4. Predictions & Core Metrics
# ----------------------------------------------------------

voting_prob_platt = platt_calibrated_model.predict_proba(X_test)[:, 1]
voting_pred_platt = (voting_prob_platt >= 0.5).astype(int)

train_accuracy = platt_calibrated_model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, voting_pred_platt)
precision = precision_score(y_test, voting_pred_platt)
recall = recall_score(y_test, voting_pred_platt)
f1 = f1_score(y_test, voting_pred_platt)


# ROC AUC
fpr, tpr, thress = roc_curve(y_test, voting_prob_platt)
auc_score = auc(fpr, tpr)

# PR-AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, voting_prob_platt)
pr_auc = auc(recall_curve, precision_curve)

# Average Precision Score
avg_precision = average_precision_score(y_test, voting_prob_platt)

# Brier Score
brier = brier_score_loss(y_test, voting_prob_platt)


# ----------------------------------------------------------
# 4A. Calibration Slope & Intercept
# ----------------------------------------------------------

probs_reshaped = voting_prob_platt.reshape(-1, 1)
calib_lr = LogisticRegression().fit(probs_reshaped, y_test)

calib_slope = calib_lr.coef_[0][0]
calib_intercept = calib_lr.intercept_[0]


# ----------------------------------------------------------
# 5. Print All Metrics
# ----------------------------------------------------------

print("========== Platt-Calibrated OVEC Model ==========")
print(f"Training Accuracy       : {train_accuracy:.4f}")
print(f"Testing Accuracy        : {test_accuracy:.4f}")
print(f"Precision               : {precision:.4f}")
print(f"Recall                  : {recall:.4f}")
print(f"F1 Score                : {f1:.4f}")
print(f"AUC-ROC                 : {auc_score:.4f}")
print(f"PR-AUC                  : {pr_auc:.4f}")
print(f"Average Precision Score : {avg_precision:.4f}")
print(f"Brier Score             : {brier:.4f}")
print(f"Calibration Intercept   : {calib_intercept:.4f}")
print(f"Calibration Slope       : {calib_slope:.4f}")
print("===================================================")


# ----------------------------------------------------------
# 6. Plot ROC Curve
# ----------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Calibrated OVEC AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve – Platt-Calibrated OVEC Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 7. Plot Calibration Curve
# ----------------------------------------------------------

prob_true, prob_pred = calibration_curve(y_test, voting_prob_platt, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated OVEC')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
plt.title("Calibration Curve – Platt Scaling")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# 8. Plot Precision–Recall (PR) Curve
# ----------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")

# Baseline = positive class ratio
baseline = np.mean(y_test)
plt.hlines(baseline, xmin=0, xmax=1, colors='gray', linestyles='--', label=f'Baseline = {baseline:.3f}')

plt.title("Precision–Recall Curve – Platt-Calibrated OVEC Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()


```

### Cell 140 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# ----------------------------------------------------------
# 7. Plot Calibration Curve (Save as PDF 300 DPI)
# ----------------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated Proposed OVEC')
plt.plot([0, 1], [0, 1], '--', color='orange', label='Perfect Calibration')
#plt.title("Calibration Curve – Platt Scaling")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.legend(loc="lower right")
plt.grid(True)

calibration_path = "/content/calibration_curve_300dpi.pdf"
plt.savefig(calibration_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# Auto-download the PDF
files.download(calibration_path)


# ----------------------------------------------------------
# 8. Plot Precision–Recall (PR) Curve (Save as PDF 300 DPI)
# ----------------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, linewidth=2,
         label=f"PR-AUC of Platt-Calibrated Proposed OVEC  = {pr_auc:.4f}")

baseline = np.mean(y_test)
plt.hlines(baseline, xmin=0, xmax=1, color='orange', linestyles='--',
           label=f'Baseline (Logistic Regression) = {baseline:.4f}')

#plt.title("Precision–Recall Curve – Platt-Calibrated NOVEC Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()

pr_curve_path = "/content/pr_curve_300dpi.pdf"
plt.savefig(pr_curve_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# Auto-download the PDF
files.download(pr_curve_path)

```

### Cell 141 — code (execution_count=None)
```python

```

### Cell 142 — code (execution_count=None)
```python
y_prob = NOVEC_prob
ece, info = expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10,
    strategy="uniform"   # or "quantile"
)

print(f"{'Bin':<3} {'Range':<15} {'Count':<7} {'Frac':<8} "
      f"{'Conf':<8} {'Acc':<8} {'|Acc-Conf|':<12} {'Contribution'}")

for i in range(len(info["bin_counts"])):
    count = info["bin_counts"][i]
    frac = info["bin_fraction"][i]
    conf = info["bin_confidence"][i]
    acc = info["bin_accuracy"][i]

    if count == 0:
        print(f"{i:<3} EMPTY")
        continue

    contrib = frac * abs(acc - conf)
    bin_range = f"[{info['bin_edges'][i]:.2f}, {info['bin_edges'][i+1]:.2f}]"

    print(f"{i:<3} {bin_range:<15} {count:<7} {frac:<8.3f} "
          f"{conf:<8.3f} {acc:<8.3f} {abs(acc-conf):<12.3f} {contrib:.5f}")

print("\nTotal ECE:", ece)

```

### Cell 143 — code (execution_count=None)
```python
print(type(ece), ece)
print(info.keys())
```

### Cell 144 — code (execution_count=None)
```python
import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(y_prob, qs)
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
        bin_edges = np.unique(bin_edges)  # avoid duplicate edges
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    ece = 0.0
    n = len(y_true)

    bin_acc, bin_conf, bin_frac, bin_counts = [], [], [], []

    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        count = mask.sum()

        if count == 0:
            bin_acc.append(np.nan)
            bin_conf.append(np.nan)
            bin_frac.append(0.0)
            bin_counts.append(0)
            continue

        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        frac = count / n

        ece += frac * abs(acc - conf)

        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_frac.append(frac)
        bin_counts.append(count)

    info = {
        "bin_edges": bin_edges,
        "bin_accuracy": np.array(bin_acc),
        "bin_confidence": np.array(bin_conf),
        "bin_fraction": np.array(bin_frac),
        "bin_counts": np.array(bin_counts),
    }

    return ece, info

def print_bin_table(info, ece, title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

    print(f"{'Bin':<3} {'Range':<15} {'Count':<7} {'Frac':<8} "
          f"{'Conf':<8} {'Acc':<8} {'|Acc-Conf|':<12} {'Contribution'}")

    for i in range(len(info["bin_counts"])):
        count = info["bin_counts"][i]
        if count == 0:
            print(f"{i:<3} EMPTY")
            continue

        frac = info["bin_fraction"][i]
        conf = info["bin_confidence"][i]
        acc = info["bin_accuracy"][i]
        contrib = frac * abs(acc - conf)
        bin_range = f"[{info['bin_edges'][i]:.2f}, {info['bin_edges'][i+1]:.2f}]"

        print(f"{i:<3} {bin_range:<15} {count:<7} {frac:<8.3f} "
              f"{conf:<8.3f} {acc:<8.3f} {abs(acc-conf):<12.3f} {contrib:.5f}")

    print("\nECE:", ece)

y_prob = NOVEC_prob   # your probabilities
# y_true = ...        # your true labels

# --- Uniform bins (diagnostics) ---
ece_u, info_u = expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10,
    strategy="uniform"
)

print_bin_table(
    info_u,
    ece_u,
    title="ECE with Uniform Bins (10) — Diagnostic"
)

# --- Quantile bins (reporting) ---
ece_q, info_q = expected_calibration_error(
    y_true,
    y_prob,
    n_bins=5,
    strategy="quantile"
)

print_bin_table(
    info_q,
    ece_q,
    title="ECE with Quantile Bins (5) — Reporting"
)

```

### Cell 145 — markdown (execution_count=None)
```text
**Calibration Curve (Uniform Bins — Diagnostic)**
```

### Cell 146 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()  # prevent duplicate rendering in notebooks

def compute_uniform_bin_stats(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    mean_pred, frac_pos, bin_counts, bin_ranges = [], [], [], []

    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue

        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
        bin_counts.append(count)
        bin_ranges.append((bin_edges[b], bin_edges[b + 1]))

    return (
        np.array(mean_pred),
        np.array(frac_pos),
        np.array(bin_counts),
        bin_ranges
    )

def smart_annotate_no_overlap(ax, x, y, text, placed, y_step=0.06):
    """
    Collision-aware annotation placement.
    Ensures labels never overlap.
    """

    # Prefer below for high y, above for low y
    dy = -0.06 if y > 0.75 else 0.06
    new_y = y + dy

    # Shift label until no collision
    while any(abs(new_y - py) < y_step and abs(x - px) < 0.12
              for px, py in placed):
        new_y += y_step if dy > 0 else -y_step

    placed.append((x, new_y))

    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, new_y),
        textcoords="data",
        ha="center",
        va="bottom" if new_y > y else "top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.95),
        arrowprops=dict(arrowstyle="->", lw=0.85)
    )

def plot_uniform_calibration_ece_aligned(
    y_true,
    y_prob,
    n_bins=10,
    filename="uniform_calibration_diagnostic_300dpi.pdf"
):
    mean_pred, frac_pos, bin_counts, bin_ranges = compute_uniform_bin_stats(
        y_true, y_prob, n_bins
    )

    fig, ax = plt.subplots(figsize=(11, 11))

    # Model curve
    ax.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2.5,
        label="Proposed OVEC Voting Model"
    )

    # Perfect calibration
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Perfect calibration"
    )

    # Collision-aware annotations
    placed_labels = []
    for x, y, c, (l, r) in zip(mean_pred, frac_pos, bin_counts, bin_ranges):
        label = f"[{l:.1f}, {r:.1f}]\n n={c}\n({x:.2f}, {y:.2f})"
        smart_annotate_no_overlap(ax, x, y, label, placed_labels)

    ax.set_xlabel("Mean predicted probability", fontsize=14)
    ax.set_ylabel("Observed fraction of positives", fontsize=14)
    #ax.set_title("Calibration Curve (Uniform Bins — Diagnostic)", fontsize=13)
    ax.legend(frameon=False, fontsize=14)
    ax.grid(alpha=0.5)

    plt.tight_layout()

    # ✅ SAVE (vector PDF, 300 DPI)
    plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {filename}")

plot_uniform_calibration_ece_aligned(
    y_true,
    NOVEC_prob,
    n_bins=10,
    filename="uniform_calibration_diagnostic_300dpi.pdf"
)

from google.colab import files
files.download("uniform_calibration_diagnostic_300dpi.pdf")

```

### Cell 147 — code (execution_count=None)
```python

```

### Cell 148 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_quantile_calibration_publication(
    y_true,
    y_prob,
    n_bins=5,
    title="Calibration Curve (Quantile Bins — Reporting)"
):
    # Compute calibration curve with quantile bins
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile"
    )

    plt.figure(figsize=(4.5, 4.5))

    # Model curve
    plt.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        label="Proposed OSEC Stacking Model"
    )

    # Perfect calibration reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration"
    )

    # Styling for publication
    plt.xlabel("Mean predicted probability", fontsize=11)
    plt.ylabel("Observed fraction of positives", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(frameon=False, fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()

plot_quantile_calibration_publication(
    y_true,
    NOVEC_prob,
    n_bins=5
)
```

### Cell 149 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_quantile_calibration_with_counts(
    y_true,
    y_prob,
    n_bins=5
    #title="Calibration Curve (Quantile Bins — Reporting)"
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Calibration curve (quantile bins)
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile"
    )

    # Compute bin counts manually
    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    bin_counts = np.array([(bin_ids == i).sum() for i in range(n_bins)])

    # Sort by x for consistent alternation
    order = np.argsort(mean_pred)
    mean_pred = mean_pred[order]
    frac_pos = frac_pos[order]
    bin_counts = bin_counts[order]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="Proposed OVEC Voting Model")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect calibration")

    # Small offset close to the line
    y_offset = 0.025

    for i, (x, y, c) in enumerate(zip(mean_pred, frac_pos, bin_counts)):
        if i % 2 == 0:
            y_text = y + y_offset
            va = "bottom"
        else:
            y_text = y - y_offset
            va = "top"

        ax.text(
            x,
            y_text,
            f"n={c}",
            ha="center",
            va=va,
            fontsize=9
        )

    # Styling
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed fraction of positives", fontsize=11)
    #ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)

    ax.set_xlim(-0.06, 1.05)
    ax.set_ylim(-0.06, 1.07)

    plt.tight_layout()
    plt.show()


# Usage
plot_quantile_calibration_with_counts(
    y_true,
    NOVEC_prob,
    n_bins=5
)

```

### Cell 150 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from google.colab import files

def plot_quantile_calibration_with_counts(
    y_true,
    y_prob,
    n_bins=5,
    filename="OVEC_calibration_curve.pdf"
):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Calibration curve (quantile bins)
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile"
    )

    # Compute bin counts manually
    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)
    bin_counts = np.array([(bin_ids == i).sum() for i in range(n_bins)])

    # Sort by x for consistent alternation
    order = np.argsort(mean_pred)
    mean_pred = mean_pred[order]
    frac_pos = frac_pos[order]
    bin_counts = bin_counts[order]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
            label="Proposed OVEC Voting Model")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5,
            label="Perfect calibration")

    # Small offset close to the line
    y_offset = 0.025
    for i, (x, y, c) in enumerate(zip(mean_pred, frac_pos, bin_counts)):
        if i % 2 == 0:
            y_text = y + y_offset
            va = "bottom"
        else:
            y_text = y - y_offset
            va = "top"

        ax.text(
            x,
            y_text,
            f"n={c}",
            ha="center",
            va=va,
            fontsize=9
        )

    # Styling
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed fraction of positives", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)

    ax.set_xlim(-0.06, 1.05)
    ax.set_ylim(-0.06, 1.07)

    plt.tight_layout()

    # ---- SAVE AS 300 DPI PDF ----
    plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # ---- AUTO DOWNLOAD (Colab) ----
    files.download(filename)


# Usage
plot_quantile_calibration_with_counts(
    y_true,
    NOVEC_prob,
    n_bins=5,
    filename="OVEC_calibration_curve_300dpi.pdf"
)

```

### Cell 151 — markdown (execution_count=None)
```text
## Decision Curve Analysis (DCA)
```

### Cell 152 — code (execution_count=None)
```python
# ============================================================
# Decision Curve Analysis (DCA) – Final Revised Code
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. DCA Function (Corrected + Clean)
# ------------------------------------------------------------
def decision_curve_analysis(y_true, y_prob,
                            thresholds=np.linspace(0.01, 0.99, 100)):
    """
    Compute net benefit across threshold probabilities.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    thresholds : array-like
        Threshold probability values for DCA.

    Returns:
    --------
    thresholds : array-like
        Thresholds used in DCA.
    net_benefit : numpy array
        Net benefit values corresponding to thresholds.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    net_benefit = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        # Standard net benefit formula:
        # NB = (TP/n) - (FP/n) * (t / (1 - t))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(nb)

    return thresholds, np.array(net_benefit)


# ------------------------------------------------------------
# 2. Inputs for NOVEC Voting Classifier
# ------------------------------------------------------------
# Replace with your actual variables:
# y_test_old : true labels
# novec_prob : predicted probabilities of NOVEC classifier

y_true = y_test

Voting_prob_platt = platt_calibrated_model.predict_proba(X_test)[:, 1]
Voting_pred_platt = (Voting_prob_platt >= 0.5).astype(int)


y_prob = Voting_prob_platt   # shape (n_samples,)


# ------------------------------------------------------------
# 3. Compute DCA Values
# ------------------------------------------------------------
thresholds, net_benefit_novec = decision_curve_analysis(y_true, y_prob)

# Treat None strategy (always zero)
nb_treat_none = np.zeros_like(thresholds)

# Treat All strategy (depends on prevalence)
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


# ------------------------------------------------------------
# 4. Plot DCA Curve
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(thresholds, net_benefit_novec, label="OVEC Voting Classifier", linewidth=2.1)
plt.plot(thresholds, nb_treat_none, '--', label="Treat None", linewidth=1.8)
plt.plot(thresholds, nb_treat_all, '--', label="Treat All", linewidth=1.8)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA) – OVEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

### Cell 153 — code (execution_count=None)
```python
from google.colab import files

# ------------------------------------------------------------
# 4. Plot & Save DCA Curve (300 DPI PDF + Auto-Download)
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(thresholds, net_benefit_novec, label="Proposed OVEC Voting Ensemble Classifier", linewidth=2.1)
plt.plot(thresholds, nb_treat_none, '--', label="Treat None", linewidth=1.8)
plt.plot(thresholds, nb_treat_all, '--', label="Treat All", linewidth=1.8)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
#plt.title("Decision Curve Analysis (DCA) – OSEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save PDF at 300 dpi
dca_pdf_path = "/content/DCA_OVEC_300dpi.pdf"
plt.savefig(dca_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# Auto-download PDF
files.download(dca_pdf_path)

```

### Cell 154 — code (execution_count=None)
```python
# ------------------------------------------------------------
# 3. Compute DCA Values
# ------------------------------------------------------------
thresholds, net_benefit_novec = decision_curve_analysis(y_true, y_prob)

# ------------------------------------------------------------
# 3A. Print Net Benefit Along With Threshold
# ------------------------------------------------------------
print("\n=== Threshold vs Net Benefit (OVEC Voting Classifier) ===")
print(f"{'Threshold':<12} {'Net Benefit':<12}")

for t, nb in zip(thresholds, net_benefit_novec):
    print(f"{t:<12.4f} {nb:<12.6f}")

# Treat None strategy (always zero)
nb_treat_none = np.zeros_like(thresholds)

# Treat All strategy (depends on prevalence)
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

```

### Cell 155 — code (execution_count=None)
```python

```

### Cell 156 — markdown (execution_count=None)
```text
## Mean ± SD (OVEC)
```

### Cell 157 — code (execution_count=None)
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

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

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



# ----------------------------------------------------------
# 3. Apply Platt Scaling (Sigmoid Calibration)
# ----------------------------------------------------------

platt_calibrated_model = CalibratedClassifierCV(
    estimator=voting_model,
    method='sigmoid',
    cv=cv,
    n_jobs=-1
)

platt_calibrated_model.fit(X_train, y_train)

```

### Cell 158 — code (execution_count=None)
```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

ys66_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys66_cv = cross_validate(
    platt_calibrated_model, X_train, y_train,
    cv=5, scoring=ys66_scoring
)

print("\n===== NOSEC (ys66) – Mean ± SD over 5-Fold CV =====")
for metric in ys66_scoring:
    scores = ys66_cv[f"test_{metric}"]
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")
print("=====================================================\n")

# ------------------------------------------------------------
# 7. Mean ± SD Plot
# ------------------------------------------------------------

ys66_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ys66_means = [
    ys66_cv['test_accuracy'].mean(),
    ys66_cv['test_precision'].mean(),
    ys66_cv['test_recall'].mean(),
    ys66_cv['test_f1'].mean(),
    ys66_cv['test_roc_auc'].mean()
]

ys66_stds = [
    ys66_cv['test_accuracy'].std(),
    ys66_cv['test_precision'].std(),
    ys66_cv['test_recall'].std(),
    ys66_cv['test_f1'].std(),
    ys66_cv['test_roc_auc'].std()
]

x = np.arange(len(ys66_metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys66_means,
    yerr=ys66_stds,
    fmt='o-', capsize=6, linewidth=2, markersize=8,
    color='darkred', ecolor='pink'
)

plt.xticks(x, ys66_metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Mean ± SD – NOSEC Stacking Classifier (ys66)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

```

### Cell 159 — code (execution_count=None)
```python
# ✅ Step 1: Define base classifiers (from your variables)
base_classifiers = {
    ('knn', knn_best),
     ('rf', rf_best),
      ('logistic', logistic_best),
       ('svm', svm_best),
        ('Voting (OVEC)', platt_calibrated_model)
}

```

### Cell 160 — code (execution_count=None)
```python
# ✅ Step 2: Scoring dictionary
ys66_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

```

### Cell 161 — code (execution_count=None)
```python
# ✅ Step 3: Cross-validate all base models + stacking
from sklearn.model_selection import cross_validate

cv_results = {}

base_classifiers = {
    "KNN": knn_best,
    "SVM": svm_best,
    "Logistic Regression": logistic_best,
    "Random Forest": rf_best,
    "Voting": platt_calibrated_model
}

for model_name, model in base_classifiers.items():
    print(f"Running 5-Fold CV for: {model_name}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=ys66_scoring,
        n_jobs=-1,
        return_train_score=False
    )

    cv_results[model_name] = scores

```

### Cell 162 — code (execution_count=None)
```python
# ✅ Step 4: Print Mean ± SD table (paper-ready)
print("\n===== Mean ± SD over 5-Fold CV =====")

for model_name, scores in cv_results.items():
    print(f"\n--- {model_name} ---")
    for metric in ys66_scoring:
        vals = scores[f"test_{metric}"]
        print(f"{metric.capitalize():<10}: {vals.mean():.4f} ± {vals.std():.4f}")

print("===================================")

```

### Cell 163 — code (execution_count=None)
```python

```

### Cell 164 — code (execution_count=None)
```python

```

### Cell 165 — code (execution_count=None)
```python

```

### Cell 166 — markdown (execution_count=None)
```text
## Prepare for FIVE STATE Test Data
```

### Cell 167 — code (execution_count=None)
```python
data = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data
data.head()
```

### Cell 168 — code (execution_count=None)
```python
print(data.columns.tolist())

```

### Cell 169 — code (execution_count=None)
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

### Cell 170 — code (execution_count=None)
```python
X = data.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data['HeartDisease']
print(data)
print("X",X)
print("y",y)

#Target Variables Values
data.HeartDisease.value_counts()
```

### Cell 171 — code (execution_count=None)
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

### Cell 172 — code (execution_count=None)
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

### Cell 173 — code (execution_count=None)
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

### Cell 174 — code (execution_count=None)
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

### Cell 175 — code (execution_count=None)
```python
# 2️⃣ Apply LoRAS
df_imputed_X_minority_loras = loras.fit_resample( df_imputed_X_majority, df_imputed_X_minority)
print("After LoRAS:")
print("Minority class:", df_imputed_X_minority_loras.shape)
print("Majority class:", df_imputed_X_majority.shape)
```

### Cell 176 — code (execution_count=None)
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

### Cell 177 — code (execution_count=None)
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

### Cell 178 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(X_loras)
```

### Cell 179 — markdown (execution_count=None)
```text
## Decision Curve Analysis (OSEC DCA)
```

### Cell 180 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, y_loras, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 181 — code (execution_count=None)
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

### Cell 182 — code (execution_count=None)
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
    "n_estimators": [300],
    #"n_estimators": [50, 100, 200, 300],
    "max_depth": [3],                 # ✅ keep shallow
   # "max_depth": [2, 3, 5],
    "learning_rate": [0.1],
    #"learning_rate": [0.01, 0.001, 0.0001],
    "subsample": [0.8],
    #"subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.9],
    #"colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1],
    #"min_child_weight": [1, 3, 5, 8],
    "gamma": [0],
    #"gamma": [0, 0.05, 0.1],
    #"reg_alpha": [0, 0.05, 0.1],
    "reg_alpha": [0],
    #"reg_lambda": [3, 5, 10],
    "reg_lambda": [5],
}
xgb_meta = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_meta, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
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
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

stacking_classifier = StackingClassifier(
    estimators=[('bayes', best_bayes), ('adaboost', best_adaboost), ('tree', best_tree), ('logistic', best_logistic)],
    final_estimator=best_xgb,
    passthrough=True,
    cv=5,
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

### Cell 183 — code (execution_count=None)
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
    "n_estimators": [300],
    #"n_estimators": [50, 100, 200, 300],
    "max_depth": [3],                 # ✅ keep shallow
   # "max_depth": [2, 3, 5],
    "learning_rate": [0.1],
    #"learning_rate": [0.01, 0.001, 0.0001],
    "subsample": [0.8],
    #"subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.9],
    #"colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1],
    #"min_child_weight": [1, 3, 5, 8],
    "gamma": [0],
    #"gamma": [0, 0.05, 0.1],
    #"reg_alpha": [0, 0.05, 0.1],
    "reg_alpha": [0],
    #"reg_lambda": [3, 5, 10],
    "reg_lambda": [5],
}
xgb_meta = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_meta, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
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
from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

stacking_classifier = StackingClassifier(
    estimators=[('bayes', best_bayes), ('adaboost', best_adaboost), ('tree', best_tree), ('logistic', best_logistic)],
    final_estimator=best_xgb,
    passthrough=True,
    cv=5,
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

### Cell 184 — code (execution_count=None)
```python
# ============================================================
# Decision Curve Analysis (DCA) – Final Revised Code
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. DCA Function (Corrected + Clean)
# ------------------------------------------------------------
def decision_curve_analysis(y_true, y_prob,
                            thresholds=np.linspace(0.01, 0.99, 100)):
    """
    Compute net benefit across threshold probabilities.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    thresholds : array-like
        Threshold probability values for DCA.

    Returns:
    --------
    thresholds : array-like
        Thresholds used in DCA.
    net_benefit : numpy array
        Net benefit values corresponding to thresholds.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    net_benefit = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        # Standard net benefit formula:
        # NB = (TP/n) - (FP/n) * (t / (1 - t))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(nb)

    return thresholds, np.array(net_benefit)


# ------------------------------------------------------------
# 2. Inputs for NOVEC Stacking Classifier
# ------------------------------------------------------------
# Replace with your actual variables:
# y_test_old : true labels
# novec_prob : predicted probabilities of NOVEC classifier

y_true = y_test

Stacking_prob_platt = calibrated_stacking.predict_proba(X_test)[:, 1]
Stacking_pred_platt = (Stacking_prob_platt >= 0.5).astype(int)


y_prob = Stacking_prob_platt   # shape (n_samples,)


# ------------------------------------------------------------
# 3. Compute DCA Values
# ------------------------------------------------------------
thresholds, net_benefit_novec = decision_curve_analysis(y_true, y_prob)

# Treat None strategy (always zero)
nb_treat_none = np.zeros_like(thresholds)

# Treat All strategy (depends on prevalence)
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


# ------------------------------------------------------------
# 4. Plot DCA Curve
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(thresholds, net_benefit_novec, label="OSEC Stacking Classifier", linewidth=2.1)
plt.plot(thresholds, nb_treat_none, '--', label="Treat None", linewidth=1.8)
plt.plot(thresholds, nb_treat_all, '--', label="Treat All", linewidth=1.8)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA) – OSEC Stacking Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

### Cell 185 — code (execution_count=None)
```python
from google.colab import files

# ------------------------------------------------------------
# 4. Plot & Save DCA Curve (300 DPI PDF + Auto-Download)
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(thresholds, net_benefit_novec, label="Proposed OSEC Stacking Ensemble Classifier", linewidth=2.1)
plt.plot(thresholds, nb_treat_none, '--', label="Treat None", linewidth=1.8)
plt.plot(thresholds, nb_treat_all, '--', label="Treat All", linewidth=1.8)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
#plt.title("Decision Curve Analysis (DCA) – OSEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save PDF at 300 dpi
dca_pdf_path = "/content/DCA_OVEC_300dpi.pdf"
plt.savefig(dca_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# Auto-download PDF
files.download(dca_pdf_path)

```

### Cell 186 — code (execution_count=None)
```python
# ------------------------------------------------------------
# 3. Compute DCA Values
# ------------------------------------------------------------
thresholds, net_benefit_novec = decision_curve_analysis(y_true, y_prob)

# ------------------------------------------------------------
# 3A. Print Net Benefit Along With Threshold
# ------------------------------------------------------------
print("\n=== Threshold vs Net Benefit (OSEC Stacking Classifier) ===")
print(f"{'Threshold':<12} {'Net Benefit':<12}")

for t, nb in zip(thresholds, net_benefit_novec):
    print(f"{t:<12.4f} {nb:<12.6f}")

# Treat None strategy (always zero)
nb_treat_none = np.zeros_like(thresholds)

# Treat All strategy (depends on prevalence)
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

```

### Cell 187 — code (execution_count=None)
```python

```

### Cell 188 — code (execution_count=None)
```python

```

### Cell 189 — markdown (execution_count=None)
```text
# Proposed OSEC on FIVE States Processed data
```

### Cell 190 — code (execution_count=None)
```python
X_test_five = X_oversampled_LoRAS_features_pca9
y_test_five = y_loras

```

### Cell 191 — code (execution_count=None)
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

### Cell 192 — markdown (execution_count=None)
```text

```

### Cell 193 — markdown (execution_count=None)
```text
# Proposed OSEC on FIVE States Original data
```

### Cell 194 — code (execution_count=None)
```python
X_five = X
y_five = y
print("X",X)
print("y",y)
```

### Cell 195 — code (execution_count=None)
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

### Cell 196 — markdown (execution_count=None)
```text
## Proposed OVEC on FIVE States Processed data
```

### Cell 197 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train_five, X_test_five, y_train_five, y_test_five = train_test_split(X_oversampled_LoRAS_features_pca9, y_loras, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train_five.shape}')
print(f'Test features shape: {X_test_five.shape}')
print(f'Train labels shape: {y_train_five.shape}')
print(f'Test labels shape: {y_test_five.shape}')

# 1. Fit the Voting Classifier
voting_model.fit(X_train_five, y_train_five)
```

### Cell 198 — code (execution_count=None)
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