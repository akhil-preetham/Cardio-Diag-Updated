# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\five_states_results\Five States Results\Wilcoxon,_Delong,_Bootstrapped.ipynb`
- SHA-256: `68b328092c641e6c06941048e06b1fbf19871cf970f5f91b1238a38f9af1eed1`
- Cells: 126
- Saved outputs: 96

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
- `sklearn.base`
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
- delong
- train_test_split
- wilcoxon

## Risk flags
- **credential_or_mount**: cells [1, 52, 74, 78, 79, 81]
- **notebook_shell**: cells [22, 62]

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
```

### Cell 5 — code (execution_count=None)
```python
X = data.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data['HeartDisease']
print(data)
print("X",X)
print("y",y)

#Target Variables Values
data.HeartDisease.value_counts()
```

### Cell 6 — markdown (execution_count=None)
```text
# **Daza Stacking 1A**
```

### Cell 7 — code (execution_count=None)
```python
# Apply oversampling using SMOTE to balance the dataset
smote = SMOTE()
X_daza, y_daza = smote.fit_resample(X, y)
```

### Cell 8 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_smote_features_pca9 = pca.fit_transform(X_daza)
```

### Cell 9 — code (execution_count=None)
```python
X_train_daza, X_test_daza, y_train_daza, y_test_daza = train_test_split(X_oversampled_smote_features_pca9 , y_daza, test_size=0.25, random_state=42, shuffle=True)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train_daza.shape}')
print(f'Test features shape: {X_test_daza.shape}')
print(f'Train labels shape: {y_train_daza.shape}')
print(f'Test labels shape: {y_test_daza.shape}')
```

### Cell 10 — code (execution_count=None)
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

### Cell 11 — code (execution_count=None)
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

### Cell 12 — code (execution_count=None)
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

### Cell 13 — code (execution_count=None)
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

### Cell 14 — code (execution_count=None)
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
plt.plot(fpr_daza, tpr_daza, color='blue', label=f'ROC curve (area_daza = {auc_daza:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

```

### Cell 15 — code (execution_count=None)
```python
len(y_test_daza) == 254

```

### Cell 16 — markdown (execution_count=None)
```text
# **Anamoly/ Outllier Dection by Percentile Method**
```

### Cell 17 — code (execution_count=None)
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

### Cell 18 — markdown (execution_count=None)
```text
# **Dealing Outliers with Data Imputation Methods **
```

### Cell 19 — code (execution_count=None)
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

### Cell 20 — code (execution_count=None)
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

### Cell 21 — markdown (execution_count=None)
```text


# Step 2: Apply Oversampling Technoques/ Data Balancing Technique LoRAS

```

### Cell 22 — code (execution_count=None)
```python
# Install LoRAS package
!pip install loras

# Import necessary libraries
import loras
import numpy as np
import pandas as pd

# Assuming you have the dataset loaded already as features (X) and labels (y)
# labels = df_imputed_X  # Use your features from the dataset
# features = df_imputed_y  # Your target (HeartDisease column)
# Use df_imputed_y for filtering, as it matches the DataFrame shape
y_filter = df_imputed_y

# Separate the majority and minority classes
# Use y_filter instead of y for filtering
X_majority_class = df_imputed[y_filter == 1]  # Majority class (HeartDisease = 1)
X_minority_class = df_imputed[y_filter == 0]  # Minority class (HeartDisease = 0)

# Apply LoRAS to oversample the minority class
X_oversampled_minority_class_loras = loras.fit_resample(X_majority_class, X_minority_class)

# Check if the oversampled minority class is larger than 508, if so, downsample it
if len(X_oversampled_minority_class_loras) > len(X_majority_class):
    # Downsample the minority class to match the size of the majority class (508)
    np.random.seed(42)  # Set a seed for reproducibility
    X_oversampled_minority_class_loras = X_oversampled_minority_class_loras[
        np.random.choice(len(X_oversampled_minority_class_loras), size=len(X_majority_class), replace=False)
    ]

# Print the new shape of the oversampled minority class
print("Adjusted X_oversampled_minority_class_loras.shape:", X_oversampled_minority_class_loras.shape)


# Concatenate the oversampled minority class with the majority class
LoRAS_features = np.concatenate((X_oversampled_minority_class_loras, X_majority_class))
LoRAS_labels = np.concatenate((np.ones(len(X_oversampled_minority_class_loras)), np.zeros(len(X_majority_class))))

# Print the shapes of the new feature set and labels
print("LoRAS_features.shape", LoRAS_features.shape)
print("LoRAS_labels.shape", LoRAS_labels.shape)
```

### Cell 23 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming LoRAS_labels contains the class labels
# Count the number of instances for each class
class_counts_loras = np.bincount(LoRAS_labels.astype(int))  # Count 0s (majority) and 1s (minority)

# Labels for the bar chart
labels = ['Normal', 'Heart Disease']  # Label names
colors = ['blue', 'green']

# Create the bar chart with separate bars
bar1 = plt.bar('Normal', class_counts_loras[0], width=0.8, edgecolor='black', color=colors[0], label='Normal')
bar2 = plt.bar('Heart Disease', class_counts_loras[1], width=0.8, edgecolor='black', color=colors[1], label='Heart Disease')

# Add a title and labels
plt.title('Class Distribution of Heart Disease After LoRAS Oversampling')
plt.xlabel('Heart Disease or Normal')
plt.ylabel('Number of Instances')

# Add numbers on top of each bar
for bar in [bar1, bar2]:
    plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 5,
             str(int(bar[0].get_height())), ha='center', fontsize=12, color='red')

# Add legend
plt.legend()

# Display the plot
plt.show()
```

### Cell 24 — markdown (execution_count=None)
```text
# **PCA to 9**
```

### Cell 25 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(LoRAS_features)
```

### Cell 26 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, LoRAS_labels, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 27 — markdown (execution_count=None)
```text
**Algorithms hyper parameter-tuning**
```

### Cell 28 — markdown (execution_count=None)
```text
# **DT**
```

### Cell 29 — code (execution_count=None)
```python
#  A Decision Tree
param_grid_tree = {
    'max_depth': [2, 4, 5, None],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

# A Decision Tree Grid Search (Base 1 Classifier)
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
best_tree = grid_search_tree.best_estimator_
# Output the best parameters
print("Best Parameters of Decision Tree Grid Search (Base 1 Classifier):", best_tree)
```

### Cell 30 — code (execution_count=None)
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_tree.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_tree = best_tree.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_tree:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_tree = cross_val_score(best_tree, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_tree = cv_scores_tree.mean()
print(f"Validation Accuracy of Decision Tree: {validation_accuracy_tree:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_tree = best_tree.predict(X_test)

# 4. Calculate other performance metrics
print("Confusion Matrix of DT:")
print(confusion_matrix(y_test, y_pred_tree))

print("Classification Report of DT:")
print(classification_report(y_test, y_pred_tree))

# Optional: If you want to calculate accuracy directly
test_accuracy = accuracy_score(y_test, y_pred_tree)
print(f"Test Accuracy of DT: {test_accuracy:.4f}")
```

### Cell 31 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys51 Classifier
best_tree.fit(X_train, y_train)

# Make predictions
ys51_predictions = best_tree.predict(X_test)

# Calculate metrics
train_accuracy_ys51 = best_tree.score(X_train, y_train)
test_accuracy_ys51 = accuracy_score(y_test, ys51_predictions)
precision_ys51 = precision_score(y_test, ys51_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys51 = recall_score(y_test, ys51_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys51 = f1_score(y_test, ys51_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys51 would be needed for using probabilities)
if hasattr(best_tree, "predict_proba"):
    ys51_probabilities = best_tree.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys51, tpr_ys51, thresholds = roc_curve(y_test, ys51_probabilities)
    auc_score = auc(fpr_ys51, tpr_ys51)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys51, tpr_ys51, color='blue', label=f'Stacking Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys51."

# Display all metrics
print(f'Training Accuracy_ys51: {train_accuracy_ys51:.4f}')
print(f'Testing Accuracy_ys51: {test_accuracy_ys51:.4f}')
print(f'Precision_ys51: {precision_ys51:.4f}')
print(f'Recall_ys51: {recall_ys51:.4f}')
print(f'F1_ys51 Score: {f1_ys51:.4f}')
print(f'AUC-ROC_ys51: {auc_score}')


# Compute confusion matrix
cm_ys51 = confusion_matrix(y_test, ys51_predictions)

# Calculate percentages
cm_ys51_percent = cm_ys51.astype('float') / cm_ys51.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys51, cm_ys51_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys51, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 32 — markdown (execution_count=None)
```text
# **Logistic Regression**
```

### Cell 33 — code (execution_count=None)
```python
# E Logistic Regression
param_grid_logistic = {
    'penalty': ['l1', 'l2', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']
}

# E Logistic Regression Grid Search (Meta Classifier)
logistic = LogisticRegression(max_iter=1000)
grid_search_logistic = GridSearchCV(logistic, param_grid_logistic, cv=5, scoring='accuracy')
grid_search_logistic.fit(X_train, y_train)
best_logistic = grid_search_logistic.best_estimator_
# Output the best parameters
print("Best Parameters of Logistic Regression Grid Search (Meta Classifier):", best_logistic)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_logistic.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_logistic = best_logistic.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_logistic:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_logistic = cross_val_score(best_logistic, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_logistic = cv_scores_logistic.mean()
print(f"Validation Accuracy of Decision logistic: {validation_accuracy_logistic:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_logistic = best_logistic.predict(X_test)

# 4. Calculate other pelogisticormance metrics
print("Confusion Matrix of logistic:")
print(confusion_matrix(y_test, y_pred_logistic))

print("Classification Report of logistic:")
print(classification_report(y_test, y_pred_logistic))

# Optional: If you want to calculate accuracy directly
test_accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Test Accuracy of logistic: {test_accuracy_logistic:.4f}")
```

### Cell 34 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys52 Classifier
best_logistic.fit(X_train, y_train)

# Make predictions
ys52_predictions = best_logistic.predict(X_test)

# Calculate metrics
train_accuracy_ys52 = best_logistic.score(X_train, y_train)
test_accuracy_ys52 = accuracy_score(y_test, ys52_predictions)
precision_ys52 = precision_score(y_test, ys52_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys52 = recall_score(y_test, ys52_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys52 = f1_score(y_test, ys52_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys52 would be needed for using probabilities)
if hasattr(best_logistic, "predict_proba"):
    ys52_probabilities = best_logistic.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys52, tpr_ys52, thresholds = roc_curve(y_test, ys52_probabilities)
    auc_score = auc(fpr_ys52, tpr_ys52)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys52, tpr_ys52, color='blue', label=f'Stacking Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys52."

# Display all metrics
print(f'Training Accuracy_ys52: {train_accuracy_ys52:.4f}')
print(f'Testing Accuracy_ys52: {test_accuracy_ys52:.4f}')
print(f'Precision_ys52: {precision_ys52:.4f}')
print(f'Recall_ys52: {recall_ys52:.4f}')
print(f'F1_ys52 Score: {f1_ys52:.4f}')
print(f'AUC-ROC_ys52: {auc_score}')


# Compute confusion matrix
cm_ys52 = confusion_matrix(y_test, ys52_predictions)

# Calculate percentages
cm_ys52_percent = cm_ys52.astype('float') / cm_ys52.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys52, cm_ys52_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys52, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 35 — markdown (execution_count=None)
```text
# **Bayesian Networks**
```

### Cell 36 — code (execution_count=None)
```python
# G Bayesian Networks
param_grid_bayes = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

# G Naive Bayes Grid Search (Base 4 Classifier)
bayes = GaussianNB()
grid_search_bayes = GridSearchCV(bayes, param_grid_bayes, cv=5, scoring='accuracy')
grid_search_bayes.fit(X_train, y_train)
best_bayes = grid_search_bayes.best_estimator_
# Output the best parameters
print("Best Parameters of Naive Bayes Grid Search (Base 4 Classifier):", best_bayes)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_bayes.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_bayes = best_bayes.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_bayes:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_bayes = cross_val_score(best_bayes, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_bayes = cv_scores_bayes.mean()
print(f"Validation Accuracy of Decision bayes: {validation_accuracy_bayes:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_bayes = best_bayes.predict(X_test)

# 4. Calculate other pebayesormance metrics
print("Confusion Matrix of bayes:")
print(confusion_matrix(y_test, y_pred_bayes))

print("Classification Report of bayes:")
print(classification_report(y_test, y_pred_bayes))

# Optional: If you want to calculate accuracy directly
test_accuracy_bayes = accuracy_score(y_test, y_pred_bayes)
print(f"Test Accuracy of bayes: {test_accuracy_bayes:.4f}")
```

### Cell 37 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys53 Classifier
best_bayes.fit(X_train, y_train)

# Make predictions
ys53_predictions = best_bayes.predict(X_test)

# Calculate metrics
train_accuracy_ys53 = best_bayes.score(X_train, y_train)
test_accuracy_ys53 = accuracy_score(y_test, ys53_predictions)
precision_ys53 = precision_score(y_test, ys53_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys53 = recall_score(y_test, ys53_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys53 = f1_score(y_test, ys53_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys53 would be needed for using probabilities)
if hasattr(best_bayes, "predict_proba"):
    ys53_probabilities = best_bayes.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys53, tpr_ys53, thresholds = roc_curve(y_test, ys53_probabilities)
    auc_score = auc(fpr_ys53, tpr_ys53)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys53, tpr_ys53, color='blue', label=f'Stacking Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys53."

# Display all metrics
print(f'Training Accuracy_ys53: {train_accuracy_ys53:.4f}')
print(f'Testing Accuracy_ys53: {test_accuracy_ys53:.4f}')
print(f'Precision_ys53: {precision_ys53:.4f}')
print(f'Recall_ys53: {recall_ys53:.4f}')
print(f'F1_ys53 Score: {f1_ys53:.4f}')
print(f'AUC-ROC_ys53: {auc_score}')


# Compute confusion matrix
cm_ys53 = confusion_matrix(y_test, ys53_predictions)

# Calculate percentages
cm_ys53_percent = cm_ys53.astype('float') / cm_ys53.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys53, cm_ys53_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys53, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 38 — markdown (execution_count=None)
```text
# **AdaBoost**
```

### Cell 39 — code (execution_count=None)
```python
# H AdaBoost
param_grid_ab = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.1, 1.0, 1.1, 1.2]
}

# H AdaBoost Grid Search
ab = AdaBoostClassifier()
grid_search_ab = GridSearchCV(ab, param_grid_ab, cv=5, scoring='accuracy')
grid_search_ab.fit(X_train, y_train)
best_ab = grid_search_ab.best_estimator_
print("Best Parameters of AdaBoost Grid Search):", best_ab)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_ab.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_ab = best_ab.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_ab:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_ab = cross_val_score(best_ab, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_ab = cv_scores_ab.mean()
print(f"Validation Accuracy of Decision ab: {validation_accuracy_ab:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_ab = best_ab.predict(X_test)

# 4. Calculate other peabormance metrics
print("Confusion Matrix of ab:")
print(confusion_matrix(y_test, y_pred_ab))

print("Classification Report of ab:")
print(classification_report(y_test, y_pred_ab))

# Optional: If you want to calculate accuracy directly
test_accuracy_ab = accuracy_score(y_test, y_pred_ab)
print(f"Test Accuracy of ab: {test_accuracy_ab:.4f}")
```

### Cell 40 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys54 Classifier
best_ab.fit(X_train, y_train)

# Make predictions
ys54_predictions = best_ab.predict(X_test)

# Calculate metrics
train_accuracy_ys54 = best_ab.score(X_train, y_train)
test_accuracy_ys54 = accuracy_score(y_test, ys54_predictions)
precision_ys54 = precision_score(y_test, ys54_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys54 = recall_score(y_test, ys54_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys54 = f1_score(y_test, ys54_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys54 would be needed for using probabilities)
if hasattr(best_ab, "predict_proba"):
    ys54_probabilities = best_ab.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys54, tpr_ys54, thresholds = roc_curve(y_test, ys54_probabilities)
    auc_score = auc(fpr_ys54, tpr_ys54)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys54, tpr_ys54, color='blue', label=f'Stacking Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys54."

# Display all metrics
print(f'Training Accuracy_ys54: {train_accuracy_ys54:.4f}')
print(f'Testing Accuracy_ys54: {test_accuracy_ys54:.4f}')
print(f'Precision_ys54: {precision_ys54:.4f}')
print(f'Recall_ys54: {recall_ys54:.4f}')
print(f'F1_ys54 Score: {f1_ys54:.4f}')
print(f'AUC-ROC_ys54: {auc_score}')


# Compute confusion matrix
cm_ys54 = confusion_matrix(y_test, ys54_predictions)

# Calculate percentages
cm_ys54_percent = cm_ys54.astype('float') / cm_ys54.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys54, cm_ys54_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys54, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 41 — markdown (execution_count=None)
```text
# **XGBoost Classifier** *Reviewer 2 Concern 18*
```

### Cell 42 — code (execution_count=None)
```python
# J XGBoost Classifier
param_grid_xgb = {
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree' : [0.6, 0.8, 1.0]
}

# J XGBoost Classifier Grid Search
xgb = XGBClassifier()
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
print("Best Parameters of XGBoost Classifier Grid Search:", best_xgb)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_xgb.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_xgb = best_xgb.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_xgb:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_xgb = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_xgb = cv_scores_xgb.mean()
print(f"Validation Accuracy of Decision xgb: {validation_accuracy_xgb:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_xgb = best_xgb.predict(X_test)

# 4. Calculate other pexgbormance metrics
print("Confusion Matrix of xgb:")
print(confusion_matrix(y_test, y_pred_xgb))

print("Classification Report of xgb:")
print(classification_report(y_test, y_pred_xgb))

# Optional: If you want to calculate accuracy directly
test_accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Test Accuracy of xgb: {test_accuracy_xgb:.4f}")
```

### Cell 43 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys55 Classifier
best_xgb.fit(X_train, y_train)

# Make predictions
ys55_predictions = best_xgb.predict(X_test)

# Calculate metrics
train_accuracy_ys55 = best_xgb.score(X_train, y_train)
test_accuracy_ys55 = accuracy_score(y_test, ys55_predictions)
precision_ys55 = precision_score(y_test, ys55_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys55 = recall_score(y_test, ys55_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys55 = f1_score(y_test, ys55_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys55 would be needed for using probabilities)
if hasattr(best_xgb, "predict_proba"):
    ys55_probabilities = best_xgb.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys55, tpr_ys55, thresholds = roc_curve(y_test, ys55_probabilities)
    auc_score = auc(fpr_ys55, tpr_ys55)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys55, tpr_ys55, color='blue', label=f'Stacking Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys55."

# Display all metrics
print(f'Training Accuracy_ys55: {train_accuracy_ys55:.4f}')
print(f'Testing Accuracy_ys55: {test_accuracy_ys55:.4f}')
print(f'Precision_ys55: {precision_ys55:.4f}')
print(f'Recall_ys55: {recall_ys55:.4f}')
print(f'F1_ys55 Score: {f1_ys55:.4f}')
print(f'AUC-ROC_ys55: {auc_score}')


# Compute confusion matrix
cm_ys55 = confusion_matrix(y_test, ys55_predictions)

# Calculate percentages
cm_ys55_percent = cm_ys55.astype('float') / cm_ys55.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys55, cm_ys55_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys55, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 44 — markdown (execution_count=None)
```text
# **ML Individual Comparison**
```

### Cell 45 — code (execution_count=None)
```python
import seaborn as sns
compar=pd.DataFrame(columns=["algorithm","Accuracy"])
compar.loc[len(compar)] = ['LogisticRegression', test_accuracy_logistic]
#compar.loc[len(compar)] = ['KNeighborsClassifier', test_accuracy_knn]
compar.loc[len(compar)] = ['DecisionTreeClassifier', test_accuracy]
#compar.loc[len(compar)] = ['RandomForestClassifier', test_accuracy_rf]
#compar.loc[len(compar)] = ['svc', test_accuracy_svc]
#compar.loc[len(compar)] = ['lightgbm', test_accuracy_lgbm]
#compar.loc[len(compar)] = ['Gradient Boosting', test_accuracy_gradien]
compar.loc[len(compar)] = ['Bayesian Networks', test_accuracy_bayes]
compar.loc[len(compar)] = ['AdaBoost', test_accuracy_ab]
#compar.loc[len(compar)] = ['CATBoost', accuracy_score]
compar.loc[len(compar)] = ['XGBoost Classifier', test_accuracy_xgb]
plt.figure(figsize=[10,5])
plt.title('ML Models Test Accuracy, Post-PCA with Nine Components and LoRAS Techniques')

# Create the barplot
ax = sns.barplot(x='Accuracy', y='algorithm', data=compar ,width=0.3, palette='Blues_d')

# Add the accuracy values at the end of each bar
for i in ax.patches:
    # Add accuracy values on each bar (i.get_width() is the width of the bar i.e., the accuracy value)
    plt.text(i.get_width() + 0.01, i.get_y() + i.get_height()/2,
             f'{i.get_width():.4f}',  # Format the accuracy value to two decimal places
             ha='center', va='center')

plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
plt.tight_layout()

# Display the plot
plt.show()
```

### Cell 46 — markdown (execution_count=None)
```text
# **New Stacking for 04 base** *Reviewer 2, Concern 22*
```

### Cell 47 — code (execution_count=None)
```python
# Create the Stacking Classifier 1 using the best estimators from Grid Search
stacking_classifier_xgb = StackingClassifier(
    cv=5,
    estimators=[
        ('bayes', best_bayes),
        ('Adaboost', best_ab),
        ('Decision tree', best_tree),
        ('Logistic Regression', best_logistic)
    ],
    final_estimator=best_xgb
)

# Calibrate the Stacking Classifier
calibrated_stacking_xgb = CalibratedClassifierCV(stacking_classifier_xgb, method='sigmoid', cv=5)
calibrated_stacking_xgb.fit(X_train, y_train)
```

### Cell 48 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys66 Classifier
calibrated_stacking_xgb.fit(X_train, y_train)

# Make predictions
ys66_predictions = calibrated_stacking_xgb.predict(X_test)

# Calculate metrics
train_accuracy_ys66 = calibrated_stacking_xgb.score(X_train, y_train)
test_accuracy_ys66 = accuracy_score(y_test, ys66_predictions)
precision_ys66 = precision_score(y_test, ys66_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys66 = recall_score(y_test, ys66_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys66 = f1_score(y_test, ys66_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys66 would be needed for using probabilities)
if hasattr(calibrated_stacking_xgb, "predict_proba"):
    ys66_probabilities = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys66, tpr_ys66, thresholds = roc_curve(y_test, ys66_probabilities)
    auc_score = auc(fpr_ys66, tpr_ys66)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys66, tpr_ys66, color='blue', label=f'NOSEC Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys66."

# Display all metrics
print(f'Training Accuracy_ys66: {train_accuracy_ys66:.4f}')
print(f'Testing Accuracy_ys66: {test_accuracy_ys66:.4f}')
print(f'Precision_ys66: {precision_ys66:.4f}')
print(f'Recall_ys66: {recall_ys66:.4f}')
print(f'F1_ys66 Score: {f1_ys66:.4f}')
print(f'AUC-ROC_ys66: {auc_score}')


# Compute confusion matrix
cm_ys66 = confusion_matrix(y_test, ys66_predictions)

# Calculate percentages
cm_ys66_percent = cm_ys66.astype('float') / cm_ys66.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys66, cm_ys66_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys66, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 49 — code (execution_count=None)
```python

```

### Cell 50 — code (execution_count=None)
```python

```

### Cell 51 — markdown (execution_count=None)
```text
# **New Voting for 04 base**
```

### Cell 52 — code (execution_count=None)
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

### Cell 53 — code (execution_count=None)
```python
data_old = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data_old
data_old.head()
```

### Cell 54 — code (execution_count=None)
```python
import pandas as pd

# Check for duplicates
duplicates = data_old[data_old.duplicated()]
print(f"Duplicate rows:\n{duplicates}")

# Remove duplicates
data_old = data_old.drop_duplicates()

# Verify that duplicates were removed
print(f"Data_oldset after removing duplicates:\n{data_old}")

```

### Cell 55 — code (execution_count=None)
```python
# Preprocessing
data_old = data_old.dropna()  # Drop rows with missing values

# Convert categorical columns into binary (0 and 1)
# Assuming the data_oldset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'

# Convert 'Sex' to 0 and 1 (e.g., Male = 1, Female = 0)
data_old['Sex'] = data_old['Sex'].map({'M': 1, 'F': 0})

# Encode 'ChestPainType' (use LabelEncoder for multi-class categorical variables)
chest_pain_encoder = LabelEncoder()
data_old['ChestPainType'] = chest_pain_encoder.fit_transform(data_old['ChestPainType'])

# Encode 'RestingECG'
resting_ecg_encoder = LabelEncoder()
data_old['RestingECG'] = resting_ecg_encoder.fit_transform(data_old['RestingECG'])

# Convert 'ExerciseAngina' to binary (e.g., Y = 1, N = 0)
data_old['ExerciseAngina'] = data_old['ExerciseAngina'].map({'Y': 1, 'N': 0})
data_old['ST_Slope'] = data_old['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': 0})
```

### Cell 56 — code (execution_count=None)
```python
X_old = data_old.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y_old = data_old['HeartDisease']
print(data_old)
print("X",X_old)
print("y",y_old)

#Target Variables Values
data_old.HeartDisease.value_counts()
```

### Cell 57 — code (execution_count=None)
```python
import numpy as np
import pandas as pd


# Define percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Detect outliers based on percentiles for each column
outliers = pd.DataFrame()

# Iterate over each feature in the dataset
for column in data_old.columns:
    # Calculate the lower and upper percentile values for the current column
    lower_bound_old = np.percentile(data_old[column], lower_percentile)
    upper_bound_old = np.percentile(data_old[column], upper_percentile)

    # Flag outliers in this column
    outliers[column] = (data_old[column] < lower_bound_old) | (data_old[column] > upper_bound_old)

# Combine the outliers across all columns
# An outlier is detected if it's flagged in any feature
outliers['is_outlier'] = outliers.any(axis=1)

# Filter the rows that are flagged as outliers
data_old_outliers = data_old[outliers['is_outlier']]
data_old_no_outliers = data_old[~outliers['is_outlier']]

# Output the result
print("Number of outliers detected:", data_old_outliers.shape[0])
print("Data with outliers removed:", data_old_no_outliers.shape)

#Target Variables Values
data_old_outliers.HeartDisease.value_counts()
```

### Cell 58 — code (execution_count=None)
```python
X_old = data_old_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y_old = data_old_no_outliers['HeartDisease']


#Target Variables Values
data_old_no_outliers.HeartDisease.value_counts()
X_old.shape
```

### Cell 59 — code (execution_count=None)
```python
import pandas as pd
import numpy as np

# Define the lower and upper percentile thresholds
lower_percentile = 1
upper_percentile = 99

# Identify numerical columns
numerical_cols_old = data_old.select_dtypes(include=[np.number]).columns

# Function to mark outliers as NaN
def mark_outliers_as_nan(df, cols, lower_perc, upper_perc):
    df_copy = df.copy()  # Create a copy of the Data_oldFrame to avoid modifying the original
    for col in cols:
        lower_bound = np.percentile(df_copy[col], lower_perc)
        upper_bound = np.percentile(df_copy[col], upper_perc)
        df_copy[col] = np.where((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound), np.nan, df_copy[col])
    return df_copy  # Return the modified Data_oldFrame

# Store the modified Data_oldFrame
data_old_with_nans = mark_outliers_as_nan(data_old, numerical_cols_old, lower_percentile, upper_percentile)

# Check the result
print("Data_old after marking outliers as NaN:")
print(data_old_with_nans.isnull().sum())
```

### Cell 60 — code (execution_count=None)
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Assuming your Outliers data_oldset after removing outliers
# The label column is assumed to be named 'HeartDisease'

# Split data_old into features and labels
X_Outliers_old = data_old_with_nans.drop('HeartDisease', axis=1)  # Features
y_Outliers_old = data_old_with_nans['HeartDisease']               # Labels

# Separate into heart disease and normal groups
df_heart_disease_old = data_old_with_nans[data_old_with_nans['HeartDisease'] == 1].copy()
df_normal_old = data_old_with_nans[data_old_with_nans['HeartDisease'] == 0].copy()

# Verify data_old before imputation
print("Shape of Heart Disease group before numerical imputation: ", df_heart_disease_old.shape)
print("Shape of Normal group before numerical imputation: ", df_normal_old.shape)

# Impute Numerical Columns
numerical_cols = X_Outliers_old.select_dtypes(include=[np.number]).columns
median_imputer = SimpleImputer(strategy='median')

df_heart_disease_old[numerical_cols] = median_imputer.fit_transform(df_heart_disease_old[numerical_cols])
df_normal_old[numerical_cols] = median_imputer.fit_transform(df_normal_old[numerical_cols])

print("Shape of Heart Disease group after numerical imputation: ", df_heart_disease_old.shape)
print("Shape of Normal group after numerical imputation: ", df_normal_old.shape)

# Combine both groups back together after imputation
df_imputed_old = pd.concat([df_heart_disease_old, df_normal_old], ignore_index=True)

print("Final Imputed Data_oldFrame: \n", df_imputed_old.head())
print("Missing values after imputation: \n", df_imputed_old.isnull().sum())
```

### Cell 61 — code (execution_count=None)
```python
df_imputed_old_X = df_imputed_old.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
df_imputed_old_y = df_imputed_old['HeartDisease']
print(df_imputed_old_X.shape)
```

### Cell 62 — code (execution_count=None)
```python
# Install LoRAS package
!pip install loras

# Import necessary libraries
import loras
import numpy as np
import pandas as pd

# Assuming you have the dataset loaded already as features_old (X) and labels_old (y)
labels_old = X_old # Use your features_old from the dataset
features_old =y_old  # Your target (HeartDisease column)

# Separate the majority and minority class_oldes
X_majority_class_old = X_old[y_old == 1]  # Majority class_old (HeartDisease = 1)
X_minority_class_old = X_old[y_old == 0]  # Minority class_old (HeartDisease = 0)

# Apply LoRAS to oversample the minority class_old
X_oversampled_minority_class_old_loras = loras.fit_resample(X_majority_class_old, X_minority_class_old)

# Check if the oversampled minority class_old is larger than 508, if so, downsample it
if len(X_oversampled_minority_class_old_loras) > len(X_majority_class_old):
    # Downsample the minority class_old to match the size of the majority class_old (508)
    np.random.seed(42)  # Set a seed for reproducibility
    X_oversampled_minority_class_old_loras = X_oversampled_minority_class_old_loras[
        np.random.choice(len(X_oversampled_minority_class_old_loras), size=len(X_majority_class_old), replace=False)
    ]

# Print the new shape of the oversampled minority class_old
print("Adjusted X_oversampled_minority_class_old_loras.shape:", X_oversampled_minority_class_old_loras.shape)


# Concatenate the oversampled minority class_old with the majority class_old
LoRAS_features_old = np.concatenate((X_oversampled_minority_class_old_loras, X_majority_class_old))
LoRAS_labels_old = np.concatenate((np.ones(len(X_oversampled_minority_class_old_loras)), np.zeros(len(X_majority_class_old))))

# Print the shapes of the new feature set and labels_old
print("LoRAS_features_old.shape", LoRAS_features_old.shape)
print("LoRAS_labels_old.shape", LoRAS_labels_old.shape)
```

### Cell 63 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)
# X and y should have shapes (253661, 21) and (253661,), respectively

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9_old = pca.fit_transform(LoRAS_features_old)
```

### Cell 64 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_oversampled_LoRAS_features_pca9_old, LoRAS_labels_old, test_size=0.25, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(LoRAS_features, LoRAS_labels, test_size=0.25, random_state=42)
# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train_old.shape}')
print(f'Test features shape: {X_test_old.shape}')
print(f'Train labels shape: {y_train_old.shape}')
print(f'Test labels shape: {y_test_old.shape}')
```

### Cell 65 — markdown (execution_count=None)
```text

```

### Cell 66 — code (execution_count=None)
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
log_rs.fit(X_train_old, y_train_old)
logistic_best_rs = log_rs.best_estimator_

# 2. Random Forest
param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [2],
    'criterion': ['entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train_old, y_train_old)
rf_best_rs = rf_rs.best_estimator_

# 3. KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['distance'],
    'metric': ['manhattan']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train_old, y_train_old)
knn_best_rs = knn_rs.best_estimator_

# 4. SVM
param_svm = {
    'C': [100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train_old, y_train_old)
svm_best_rs = svm_rs.best_estimator_

# Combine tuned models into VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ('knn', knn_best_rs),
        ('rf', rf_best_rs),
        ('logistic', logistic_best_rs),
        ('svm', svm_best_rs)
    ],
    voting='soft',
    weights=[4, 1, 1, 4]  # Example weights
)

# Fit the model
voting_model.fit(X_train_old, y_train_old)

# Make predictions
voting_predictions = voting_model.predict(X_test_old)

# Evaluate Metrics
train_accuracy_novec = voting_model.score(X_train_old, y_train_old)
test_accuracy_novec = accuracy_score(y_test_old, voting_predictions)
precision_novec = precision_score(y_test_old, voting_predictions)
recall_novec = recall_score(y_test_old, voting_predictions)
f1_novec = f1_score(y_test_old, voting_predictions)

# AUC-ROC Curve
voting_probabilities = voting_model.predict_proba(X_test_old)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_old, voting_probabilities)
auc_score_novec = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'NOVEC Classifier AUC = {auc_score_novec:.4f}')
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
print(f"Training Accuracy: {train_accuracy_novec:.4f}")
print(f"Testing Accuracy : {test_accuracy_novec:.4f}")
print(f"Precision         : {precision_novec:.4f}")
print(f"Recall            : {recall_novec:.4f}")
print(f"F1 Score          : {f1_novec:.4f}")
print(f"AUC-ROC           : {auc_score_novec:.4f}")
print("===========================================================")
```

### Cell 67 — code (execution_count=None)
```python
# Combine tuned models into VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ('knn', knn_best_rs),
        ('rf', rf_best_rs),
        ('logistic', logistic_best_rs),
        ('svm', svm_best_rs)
    ],
    voting='soft',
    weights=[4, 1, 1, 4]  # Example weights
)

# Fit the model
voting_model.fit(X_train_old, y_train_old)

# Make predictions
voting_predictions = voting_model.predict(X_test_old)

# Evaluate Metrics
train_accuracy_novec = voting_model.score(X_train_old, y_train_old)
test_accuracy_novec = accuracy_score(y_test_old, voting_predictions)
precision_novec = precision_score(y_test_old, voting_predictions)
recall_novec = recall_score(y_test_old, voting_predictions)
f1_novec = f1_score(y_test_old, voting_predictions)

# AUC-ROC Curve
voting_probabilities = voting_model.predict_proba(X_test_old)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_old, voting_probabilities)
auc_score_novec = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'NOVEC Classifier AUC = {auc_score_novec:.4f}')
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
print(f"Training Accuracy: {train_accuracy_novec:.4f}")
print(f"Testing Accuracy : {test_accuracy_novec:.4f}")
print(f"Precision         : {precision_novec:.4f}")
print(f"Recall            : {recall_novec:.4f}")
print(f"F1 Score          : {f1_novec:.4f}")
print(f"AUC-ROC           : {auc_score_novec:.4f}")
print("===========================================================")
```

### Cell 68 — code (execution_count=None)
```python

```

### Cell 69 — code (execution_count=None)
```python

```

### Cell 70 — code (execution_count=None)
```python

```

### Cell 71 — code (execution_count=None)
```python

```

### Cell 72 — code (execution_count=None)
```python
import numpy as np

# --- NOVEC Probabilities ---
voting_probabilities = voting_model.predict_proba(X_test_old)[:, 1]
fpr_novec, tpr_novec, _ = roc_curve(y_test_old, voting_probabilities)
auc_novec = auc(fpr_novec, tpr_novec)

# --- NOSEC Probabilities ---
ys66_probabilities = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]
fpr_nosec, tpr_nosec, _ = roc_curve(y_test, ys66_probabilities)
auc_nosec = auc(fpr_nosec, tpr_nosec)

# Select marker positions (every 15th point)
marker_idx_novec = np.linspace(0, len(fpr_novec) - 1, 10, dtype=int)
marker_idx_nosec = np.linspace(0, len(fpr_nosec) - 1, 10, dtype=int)

# --- Plot both ROC curves ---
plt.figure(figsize=(8, 6))

# NOVEC: Blue solid line with circle markers
plt.plot(fpr_novec, tpr_novec,
         color='blue', linestyle='-', linewidth=2,
         label=f'NOVEC (Voting) AUC = {auc_novec:.4f}')
plt.plot(fpr_novec[marker_idx_novec], tpr_novec[marker_idx_novec],
         'o', color='blue', markersize=6)

# NOSEC: Red dashed line with square markers
plt.plot(fpr_nosec, tpr_nosec,
         color='red', linestyle='--', linewidth=2,
         label=f'NOSEC (Stacking) AUC = {auc_nosec:.4f}')
plt.plot(fpr_nosec[marker_idx_nosec], tpr_nosec[marker_idx_nosec],
         's', color='red', markersize=6)

# Random classifier
plt.plot([0, 1], [0, 1], linestyle=':', color='gray', linewidth=1.5)

# Labels & Legend
plt.title('ROC Curve Comparison: NOVEC vs NOSEC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

```

### Cell 73 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# --- NOVEC Probabilities ---
voting_probabilities = voting_model.predict_proba(X_test_old)[:, 1]
fpr_novec, tpr_novec, _ = roc_curve(y_test_old, voting_probabilities)
auc_novec = auc(fpr_novec, tpr_novec)

# --- NOSEC Probabilities ---
ys66_probabilities = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]
fpr_nosec, tpr_nosec, _ = roc_curve(y_test, ys66_probabilities)
auc_nosec = auc(fpr_nosec, tpr_nosec)

# Marker points
marker_idx_novec = np.linspace(0, len(fpr_novec) - 1, 10, dtype=int)
marker_idx_nosec = np.linspace(0, len(fpr_nosec) - 1, 10, dtype=int)

# --- Plot ---
plt.figure(figsize=(8, 6))

# NOVEC
plt.plot(fpr_novec, tpr_novec, color='blue', linestyle='-', linewidth=2,
         label=f'NOVEC (Voting) AUC = {auc_novec:.4f}')
plt.plot(fpr_novec[marker_idx_novec], tpr_novec[marker_idx_novec],
         'o', color='blue', markersize=6)

# NOSEC
plt.plot(fpr_nosec, tpr_nosec, color='red', linestyle='--', linewidth=2,
         label=f'NOSEC (Stacking) AUC = {auc_nosec:.4f}')
plt.plot(fpr_nosec[marker_idx_nosec], tpr_nosec[marker_idx_nosec],
         's', color='red', markersize=6)

# Random line
plt.plot([0, 1], [0, 1], linestyle=':', color='gray')

#plt.title('ROC Curve Comparison: NOVEC vs NOSEC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save as PDF
pdf_path = "ROC_NOVEC_NOSEC.pdf"
plt.savefig(pdf_path, format="pdf")
plt.close()

pdf_path

```

### Cell 74 — code (execution_count=None)
```python
from google.colab import files
files.download("ROC_NOVEC_NOSEC.pdf")

```

### Cell 75 — code (execution_count=None)
```python

```

### Cell 76 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# -------------------------------------------------------------
# Compute ROC curves (Existing 5 models)
# -------------------------------------------------------------
fpr_ys51, tpr_ys51, _ = roc_curve(y_test, ys51_probabilities)   # Decision Tree
fpr_ys53, tpr_ys53, _ = roc_curve(y_test, ys53_probabilities)   # Naive Bayes
fpr_ys54, tpr_ys54, _ = roc_curve(y_test, ys54_probabilities)   # AdaBoost
fpr_ys55, tpr_ys55, _ = roc_curve(y_test, ys55_probabilities)   # XGBoost
fpr_ys57, tpr_ys57, _ = roc_curve(y_test, ys57_probabilities)   # CatBoost

auc_ys51 = auc(fpr_ys51, tpr_ys51)
auc_ys53 = auc(fpr_ys53, tpr_ys53)
auc_ys54 = auc(fpr_ys54, tpr_ys54)
auc_ys55 = auc(fpr_ys55, tpr_ys55)
auc_ys57 = auc(fpr_ys57, tpr_ys57)

# -------------------------------------------------------------
# Proposed NOVEC
# -------------------------------------------------------------
# --- NOVEC Probabilities ---
voting_probabilities = voting_model.predict_proba(X_test_old)[:, 1]
fpr_novec, tpr_novec, _ = roc_curve(y_test_old, voting_probabilities)
auc_novec = auc(fpr_novec, tpr_novec)

# -------------------------------------------------------------
# Proposed NOSEC
# -------------------------------------------------------------
nosec_prob = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]
fpr_nosec, tpr_nosec, _ = roc_curve(y_test, nosec_prob)
auc_nosec = auc(fpr_nosec, tpr_nosec)

# -------------------------------------------------------------
# Plot ROC Curves (All Models)
# -------------------------------------------------------------
plt.figure(figsize=(11, 9))

plt.plot(fpr_ys51, tpr_ys51, linestyle='-', marker='o', markevery=15,
         label=f"Decision Tree (AUC = {auc_ys51:.4f})")

plt.plot(fpr_ys53, tpr_ys53, linestyle='--', marker='s', markevery=15,
         label=f"Naive Bayes (AUC = {auc_ys53:.4f})")

plt.plot(fpr_ys54, tpr_ys54, linestyle='-.', marker='^', markevery=15,
         label=f"AdaBoost (AUC = {auc_ys54:.4f})")

plt.plot(fpr_ys55, tpr_ys55, linestyle=':', marker='d', markevery=15,
         label=f"XGBoost (AUC = {auc_ys55:.4f})")

plt.plot(fpr_ys57, tpr_ys57, linestyle='-', marker='x', markevery=15,
         label=f"CatBoost (AUC = {auc_ys57:.4f})")

# --- Proposed NOVEC (unique style)
plt.plot(fpr_novec, tpr_novec, linestyle='--', marker='P', markevery=15,
         label=f"Proposed OVEC (Optimized Voting Ensemble Classifier) (AUC = {auc_novec:.4f})")

# --- Proposed NOSEC (unique style)
plt.plot(fpr_nosec, tpr_nosec, linestyle='-', marker='*', markevery=15,
         label=f"Proposed OSEC (Optimized Stacking Ensemble Classifier) (AUC = {auc_nosec:.4f})")

# Random Chance baseline
plt.plot([0, 1], [0, 1], linestyle="--", color="black")

plt.xlabel("False Positive Rate (FPR)", fontsize=15)
plt.ylabel("True Positive Rate (TPR)", fontsize=15)
plt.title("Combined ROC Curve – Baseline Models vs Proposed NOVEC & NOSEC", fontsize=15)

plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

```

### Cell 77 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# Set a clean, professional style
sns.set(style="whitegrid", font_scale=1.3)
plt.figure(figsize=(12, 9))

# Custom color palette for readability
# Custom color palette for readability and high contrast
colors = {
    "DT": "#1f77b4",
    "NB": "#ff7f0e",
    "ADA": "#2ca02c",
    "XGB": "#d62728",
    "CAT": "#9467bd",

    # 🚀 High-contrast colors for proposed models:
    #"NOVEC": "#ff0000",   # Bright Red
    #"NOSEC": "#0047ff"    # Electric Deep Blue
    "NOVEC": "#ff005d",   # Neon Pink-Red
    "NOSEC": "#00b7ff"    # Neon Cyan
}

# Plot function to avoid repetition
def plot_roc(fpr, tpr, auc_value, label, color, linestyle, marker):
    plt.plot(
        fpr, tpr,
        linewidth=2.4,
        linestyle=linestyle,
        marker=marker,
        markersize=6,
        markevery=60,
        color=color,
        label=f"{label} (AUC = {auc_value:.4f})"
    )
    # Optional filled area for smoother perception
    #plt.fill_between(fpr, tpr, alpha=0.08, color=color)

# -------------------------------------------------------------
# Plot all ROC curves with enhanced styling
# -------------------------------------------------------------
plot_roc(fpr_ys51, tpr_ys51, auc_ys51, "Decision Tree", colors["DT"], "-", "o")
plot_roc(fpr_ys53, tpr_ys53, auc_ys53, "Naive Bayes", colors["NB"], "--", "s")
plot_roc(fpr_ys54, tpr_ys54, auc_ys54, "AdaBoost", colors["ADA"], "-.", "^")
plot_roc(fpr_ys55, tpr_ys55, auc_ys55, "XGBoost", colors["XGB"], ":", "D")
plot_roc(fpr_ys57, tpr_ys57, auc_ys57, "CatBoost", colors["CAT"], "-", "x")

plot_roc(fpr_novec, tpr_novec, auc_novec, "Proposed NOVEC", colors["NOVEC"], "--", "P")
plot_roc(fpr_nosec, tpr_nosec, auc_nosec, "Proposed NOSEC", colors["NOSEC"], "-", "*")

# Random baseline
plt.plot([0, 1], [0, 1], "k--", linewidth=1.3, alpha=0.6)

# -------------------------------------------------------------
# Aesthetic Enhancements
# -------------------------------------------------------------
plt.title("Combined ROC Curve – Baseline Models vs Proposed NOVEC & NOSEC",
          fontsize=18, fontweight="bold")
plt.xlabel("False Positive Rate (FPR)", fontsize=15)
plt.ylabel("True Positive Rate (TPR)", fontsize=15)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.02])

plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.show()

```

### Cell 78 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from google.colab import files

# Set a clean, professional style
sns.set(style="whitegrid", font_scale=1.3)
plt.figure(figsize=(12, 10))

# Custom color palette for readability and high contrast
colors = {
    "DT": "#1f77b4",
    "NB": "#ff7f0e",
    "ADA": "#2ca02c",
    "XGB": "#d62728",
    "CAT": "#9467bd",
    "NOVEC": "#ff005d",   # Neon Pink-Red
    "NOSEC": "#00b7ff"    # Neon Cyan
}

# Plot function to avoid repetition
def plot_roc(fpr, tpr, auc_value, label, color, linestyle, marker):
    plt.plot(
        fpr, tpr,
        linewidth=2.4,
        linestyle=linestyle,
        marker=marker,
        markersize=6,
        markevery=60,
        color=color,
        label=f"{label} (AUC = {auc_value:.4f})"
    )

# -------------------------------------------------------------
# Plot all ROC curves with enhanced styling
# -------------------------------------------------------------
plot_roc(fpr_ys51, tpr_ys51, auc_ys51, "Decision Tree", colors["DT"], "-", "o")
plot_roc(fpr_ys53, tpr_ys53, auc_ys53, "Naive Bayes", colors["NB"], "--", "s")
plot_roc(fpr_ys54, tpr_ys54, auc_ys54, "AdaBoost", colors["ADA"], "-.", "^")
plot_roc(fpr_ys55, tpr_ys55, auc_ys55, "XGBoost", colors["XGB"], ":", "D")
plot_roc(fpr_ys57, tpr_ys57, auc_ys57, "CatBoost", colors["CAT"], "-", "x")

plot_roc(fpr_novec, tpr_novec, auc_novec, "Proposed NOVEC", colors["NOVEC"], "--", "P")
plot_roc(fpr_nosec, tpr_nosec, auc_nosec, "Proposed NOSEC", colors["NOSEC"], "-", "*")

# Random baseline
plt.plot([0, 1], [0, 1], "k--", linewidth=1.3, alpha=0.6)

# -------------------------------------------------------------
# Aesthetic Enhancements
# -------------------------------------------------------------
#plt.title("Combined ROC Curve – Baseline Models vs Proposed NOVEC & NOSEC",fontsize=18, fontweight="bold")
#plt.title('ROC Curve Comparison: NOVEC vs NOSEC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save as PDF
pdf_path = "ROC_NOVEC_NOSEC.pdf"
plt.savefig(pdf_path,  dpi=300, format="pdf")
plt.close()

pdf_path


```

### Cell 79 — code (execution_count=None)
```python
from google.colab import files
files.download("ROC_NOVEC_NOSEC.pdf")
```

### Cell 80 — code (execution_count=None)
```python

```

### Cell 81 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from google.colab import files

# Set a clean, professional style
sns.set(style="whitegrid", font_scale=1.4)
plt.figure(figsize=(12, 10))

# Custom color palette for readability and high contrast
colors = {
    "DT": "#1f77b4",
    "NB": "#ff7f0e",
    "ADA": "#2ca02c",
    "XGB": "#d62728",
    "CAT": "#9467bd",
    "NOVEC": "#ff005d",   # Neon Pink-Red
    "NOSEC": "#00b7ff"    # Neon Cyan
}

# Plot function to avoid repetition
def plot_roc(fpr, tpr, auc_value, label, color, linestyle, marker):
    plt.plot(
        fpr, tpr,
        linewidth=2.6,
        linestyle=linestyle,
        marker=marker,
        markersize=8,
        markevery=40,
        color=color,
        label=f"{label} (AUC = {auc_value:.4f})"
    )

# -------------------------------------------------------------
# Plot all ROC curves with enhanced styling
# -------------------------------------------------------------
plot_roc(fpr_ys51, tpr_ys51, auc_ys51, "Decision Tree", colors["DT"], "-", "o")
plot_roc(fpr_ys53, tpr_ys53, auc_ys53, "Naive Bayes", colors["NB"], "--", "s")
plot_roc(fpr_ys54, tpr_ys54, auc_ys54, "AdaBoost", colors["ADA"], "-.", "^")
plot_roc(fpr_ys55, tpr_ys55, auc_ys55, "XGBoost", colors["XGB"], ":", "D")
plot_roc(fpr_ys57, tpr_ys57, auc_ys57, "CatBoost", colors["CAT"], "-", "x")

plot_roc(fpr_novec, tpr_novec, auc_novec, "Proposed Optimized Voting Ensemble Classifier", colors["NOVEC"], "--", "P")
plot_roc(fpr_nosec, tpr_nosec, auc_nosec, "Proposed Optimized Stacking Ensemble Classifier", colors["NOSEC"], "-", "*")

# Random baseline
plt.plot([0, 1], [0, 1], "k--", linewidth=1.4, alpha=0.6)

# -------------------------------------------------------------
# Aesthetic Enhancements
# -------------------------------------------------------------
# plt.title("Combined ROC Curve – Baseline Models vs Proposed NOVEC & NOSEC",
#          fontsize=18, fontweight="bold")
plt.xlabel("False Positive Rate (FPR)", fontsize=15)
plt.ylabel("True Positive Rate (TPR)", fontsize=15)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.02])

plt.legend(loc="lower right", fontsize=13, frameon=True, shadow=True)
plt.grid(alpha=0.25)
plt.tight_layout()

#plt.show()

# -------------------------------------------------------------
# SAVE AS HIGH-RES PDF (300 DPI) & AUTO-DOWNLOAD
# -------------------------------------------------------------
file_name = "roc_curve_plot.pdf"
plt.savefig(file_name, dpi=300, format="pdf", bbox_inches="tight")

# Auto download in Google Colab
files.download(file_name)

```

### Cell 82 — markdown (execution_count=None)
```text
# paired Wilcoxon across folds) and CIs for AUC/F1.
```

### Cell 83 — code (execution_count=None)
```python
# Step 1 — Run 5-fold CV for BOTH models (OVEC & OSEC)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

auc_ovec, auc_osec = [], []
f1_ovec, f1_osec = [], []

for train_idx, test_idx in kf.split(X_train_old, y_train_old):

    # Split fold
    X_tr, X_te = X_train_old[train_idx], X_train_old[test_idx]
    y_tr, y_te = y_train_old[train_idx], y_train_old[test_idx]

    # Train OVEC
    voting_model.fit(X_tr, y_tr)
    y_prob_ovec = voting_model.predict_proba(X_te)[:, 1]
    y_pred_ovec = (y_prob_ovec >= 0.5).astype(int)

    # Train OSEC
    calibrated_stacking_xgb.fit(X_tr, y_tr)
    y_prob_osec = calibrated_stacking_xgb.predict_proba(X_te)[:, 1]
    y_pred_osec = (y_prob_osec >= 0.5).astype(int)

    # Compute AUC
    auc_ovec.append(roc_auc_score(y_te, y_prob_ovec))
    auc_osec.append(roc_auc_score(y_te, y_prob_osec))

    # Compute F1
    f1_ovec.append(f1_score(y_te, y_pred_ovec))
    f1_osec.append(f1_score(y_te, y_pred_osec))

print("OVEC AUC folds :", auc_ovec)
print("OSEC AUC folds :", auc_osec)
print("OVEC F1 folds  :", f1_ovec)
print("OSEC F1 folds  :", f1_osec)

```

### Cell 84 — code (execution_count=None)
```python
# Step 2 — Compute 95% Confidence Intervals (Bootstrap)
def bootstrap_ci(values, alpha=0.05):
    lower = np.percentile(values, 100 * (alpha / 2))
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return (lower, upper)

auc_ovec_ci = bootstrap_ci(auc_ovec)
auc_osec_ci = bootstrap_ci(auc_osec)

f1_ovec_ci = bootstrap_ci(f1_ovec)
f1_osec_ci = bootstrap_ci(f1_osec)

print("\n95% CI for AUC (OVEC):", auc_ovec_ci)
print("95% CI for AUC (OSEC):", auc_osec_ci)
print("95% CI for F1  (OVEC):", f1_ovec_ci)
print("95% CI for F1  (OSEC):", f1_osec_ci)

```

### Cell 85 — code (execution_count=None)
```python
# Step 3 — Paired Wilcoxon Significance Test (OVEC vs OSEC)
from scipy.stats import wilcoxon

def safe_wilcoxon(a, b):
    diff = np.array(a) - np.array(b)
    if np.all(diff == 0):
        return "No difference across folds — Wilcoxon cannot be computed."
    stat, p = wilcoxon(a, b)
    return stat, p

print("\n=== Wilcoxon Significance Tests (OVEC vs OSEC) ===")
print("AUC Wilcoxon:", safe_wilcoxon(auc_ovec, auc_osec))
print("F1  Wilcoxon:", safe_wilcoxon(f1_ovec, f1_osec))

```

### Cell 86 — code (execution_count=None)
```python

```

### Cell 87 — code (execution_count=None)
```python

```

### Cell 88 — markdown (execution_count=None)
```text

```

### Cell 89 — code (execution_count=None)
```python
#============================================================
#0. IMPORTS
#============================================================
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, norm
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss,
    roc_curve, accuracy_score, classification_report
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.base import clone

from xgboost import XGBClassifier

```

### Cell 90 — code (execution_count=None)
```python
#============================================================
#     1. TRAIN/DEFINE ALL BASE MODELS
#============================================================

# OVEC model must be UNFITTED here:
def make_OVEC():
    return clone(voting_model)

#Train XGBoost (baseline)
param_grid_xgb = {
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

grid = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_xgb = grid.best_estimator_

# XGB factory:
def make_XGB():
    return clone(best_xgb)
```

### Cell 91 — code (execution_count=None)
```python
#============================================================
#     2. DEFINE FACTORIES FOR STACKING MODELS
#============================================================
#  DAZA stacked model factory:

def make_DAZA():
    stacking_model = StackingClassifier(
        estimators=[
            ('rf',     clone(best_rf_daza)),
            ('tree',   clone(best_tree_daza)),
            ('grad',   clone(best_gradien_daza)),
            ('bayes',  clone(best_bayes_daza)),
        ],
        final_estimator=clone(best_logistic_daza),
        cv=5
    )
    return CalibratedClassifierCV(stacking_model, cv=5, method='sigmoid')
```

### Cell 92 — code (execution_count=None)
```python
# OSEC stacking model factory:
def make_OSEC():
    stacking_model = StackingClassifier(
        estimators=[
            ('bayes', clone(best_bayes)),
            ('ab',    clone(best_ab)),
            ('tree',  clone(best_tree)),
            ('lr',    clone(best_logistic)),
        ],
        final_estimator=clone(best_xgb),
        cv=5
    )
    return CalibratedClassifierCV(stacking_model, cv=5, method='sigmoid')
```

### Cell 93 — code (execution_count=None)
```python
#============================================================
# 3. CLEAN CROSS-VALIDATION FUNCTION
#============================================================
from sklearn.model_selection import RepeatedKFold

def cross_val_metrics(make_model, X, y, cv=5):
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    #kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    aucs = []
    f1s = []

    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        model = make_model()       # ← fresh, unfitted model each fold
        model.fit(Xtr, ytr)

        prob = model.predict_proba(Xte)[:, 1]

        aucs.append(roc_auc_score(yte, prob))
        f1s.append(f1_score(yte, (prob >= 0.5)))

    return np.array(aucs), np.array(f1s)

```

### Cell 94 — code (execution_count=None)
```python
#============================================================
# 4. COMPUTE CROSS-VALIDATION METRICS FOR EACH MODEL
#============================================================
model_factories = {
    "OVEC": make_OVEC,
    "OSEC": make_OSEC,
    "1A Stacking (Daza et al., 2024)": make_DAZA,
    "XGBoost": make_XGB,
}

models_auc_cv = {}
models_f1_cv = {}

for name, maker in model_factories.items():
    aucs, f1s = cross_val_metrics(maker, X_train, y_train)
    models_auc_cv[name] = aucs
    models_f1_cv[name]  = f1s

    print(f"{name} AUCs:", aucs)
    print(f"{name} F1s :", f1s)

```

### Cell 95 — code (execution_count=None)
```python
#============================================================
#   5. TEST SET PREDICTIONS (AFTER CV)
#============================================================

prob_ovec = make_OVEC().fit(X_train, y_train).predict_proba(X_test)[:,1]
prob_osec = make_OSEC().fit(X_train, y_train).predict_proba(X_test)[:,1]
prob_daza = make_DAZA().fit(X_train, y_train).predict_proba(X_test)[:,1]
prob_xgb  = make_XGB().fit(X_train, y_train).predict_proba(X_test)[:,1]

y_true = y_test

```

### Cell 96 — code (execution_count=None)
```python
#============================================================
# 6. WILCOXON TEST
#============================================================

def safe_wilcoxon(a, b):
    if np.all(a == b):
        return "(identical across folds)"
    return wilcoxon(a, b)

```

### Cell 97 — code (execution_count=None)
```python
pairs = [
    ("OVEC", "OSEC"),
    ("OVEC", "1A Stacking (Daza et al., 2024)"),
    ("OSEC", "1A Stacking (Daza et al., 2024)"),
    ("OVEC", "XGBoost"),
    ("OSEC", "XGBoost"),
]

print("\n=== Wilcoxon Tests ===")
for m1, m2 in pairs:
    print(m1, "vs", m2, safe_wilcoxon(models_auc_cv[m1], models_auc_cv[m2]))

```

### Cell 98 — code (execution_count=None)
```python

```

### Cell 99 — code (execution_count=None)
```python
import numpy as np
from scipy.stats import norm


# ============================================================
# Midrank function (used by DeLong)
# ============================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)

    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j

    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


# ============================================================
# Fast DeLong method for two correlated ROC curves
# ============================================================
def fast_delong(predictions_sorted_transposed, label_1_count):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    tx = np.apply_along_axis(compute_midrank, 1, pos)
    ty = np.apply_along_axis(compute_midrank, 1, neg)
    tz = np.apply_along_axis(compute_midrank, 1, predictions_sorted_transposed)

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)

    v01 = tx - tx.mean(axis=1, keepdims=True)
    v10 = ty - ty.mean(axis=1, keepdims=True)

    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s


# ============================================================
# User-facing DeLong test function
# ============================================================
def delong_test(y_true, pred1, pred2):
    # Clean and validate inputs
    y_true = np.array(y_true).astype(int).flatten()
    pred1 = np.array(pred1).astype(float).flatten()
    pred2 = np.array(pred2).astype(float).flatten()

    assert len(y_true) == len(pred1) == len(pred2), \
        "Lengths of y_true, pred1, pred2 must match."

    # Sort by decreasing pred1 for DeLong's method
    order = np.argsort(-pred1)
    y = y_true[order]
    preds = np.vstack([pred1[order], pred2[order]])

    label_1_count = int(np.sum(y))

    # Safety checks
    if label_1_count == 0 or label_1_count == len(y):
        raise ValueError("y_true must contain both positive and negative classes.")

    aucs, cov = fast_delong(preds, label_1_count)

    auc1 = aucs[0]
    auc2 = aucs[1]
    diff = auc1 - auc2

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        raise ValueError("Computed variance is non-positive. Check your predictions.")

    z = diff / np.sqrt(var)
    p = 2 * norm.sf(abs(z))

    return auc1, auc2, diff, z, p

```

### Cell 100 — code (execution_count=None)
```python
# ✅ STEP 2 — Pairwise DeLong Comparisons

# Assuming you have already computed:
# prob_ovec
# prob_osec
# prob_daza
# prob_xgb
# y_true
```

### Cell 101 — code (execution_count=None)
```python
pairs = [
    ("OVEC", prob_ovec, "OSEC", prob_osec),
    ("OVEC", prob_ovec, "DAZA", prob_daza),
    ("OSEC", prob_osec, "DAZA", prob_daza),
    ("OVEC", prob_ovec, "XGBoost", prob_xgb),
    ("OSEC", prob_osec, "XGBoost", prob_xgb),
    ("DAZA", prob_daza, "XGBoost", prob_xgb),
]

print("\n=== DeLong AUC Comparisons ===")
for n1, p1, n2, p2 in pairs:
    auc1, auc2, diff, z, p = delong_test(y_true, p1, p2)
    print(f"\n{n1} vs {n2}")
    print(f"AUC {n1}: {auc1:.4f}")
    print(f"AUC {n2}: {auc2:.4f}")
    print(f"Difference: {diff:.4f}")
    print(f"z = {z:.4f}")
    print(f"p = {p:.4f}")

```

### Cell 102 — code (execution_count=None)
```python

```

### Cell 103 — code (execution_count=None)
```python

```

### Cell 104 — code (execution_count=None)
```python
def cross_val_metrics(make_model, X, y, n_splits=5, n_repeats=1):
    kf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42
    )

    aucs = []
    f1s = []

    for train_idx, test_idx in kf.split(X, y):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]

        # Create fresh model for this fold
        model_cv = make_model()

        model_cv.fit(X_train_cv, y_train_cv)
        prob = model_cv.predict_proba(X_test_cv)[:, 1]
        pred = (prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_test_cv, prob))
        f1s.append(f1_score(y_test_cv, pred))

    return np.array(aucs), np.array(f1s)

```

### Cell 105 — code (execution_count=None)
```python
from sklearn.model_selection import RepeatedStratifiedKFold

models_auc_cv = {}
models_f1_cv = {}

for name, maker in model_factories.items():
    aucs, f1s = cross_val_metrics(maker, X_train, y_train, n_splits=5, n_repeats=10)
    models_auc_cv[name] = aucs
    models_f1_cv[name]  = f1s

    print(f"{name} AUCs:", aucs)
    print(f"{name} F1s :", f1s)

```

### Cell 106 — code (execution_count=None)
```python
pairs = [
    ("OVEC", "OSEC"),
    ("OVEC", "1A Stacking (Daza et al., 2024)"),
    ("OSEC", "1A Stacking (Daza et al., 2024)"),
    ("OVEC", "XGBoost"),
    ("OSEC", "XGBoost"),
]

print("\n=== Wilcoxon Tests ===")
for m1, m2 in pairs:
    print(m1, "vs", m2, safe_wilcoxon(models_auc_cv[m1], models_auc_cv[m2]))
```

### Cell 107 — code (execution_count=None)
```python

```

### Cell 108 — code (execution_count=None)
```python
import numpy as np
from scipy.stats import norm


# ============================================================
# Midrank function (used by DeLong)
# ============================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)

    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j

    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


# ============================================================
# Fast DeLong method for two correlated ROC curves
# ============================================================
def fast_delong(predictions_sorted_transposed, label_1_count):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m

    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    tx = np.apply_along_axis(compute_midrank, 1, pos)
    ty = np.apply_along_axis(compute_midrank, 1, neg)
    tz = np.apply_along_axis(compute_midrank, 1, predictions_sorted_transposed)

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)

    v01 = tx - tx.mean(axis=1, keepdims=True)
    v10 = ty - ty.mean(axis=1, keepdims=True)

    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s


# ============================================================
# User-facing DeLong test function
# ============================================================
def delong_test(y_true, pred1, pred2):
    # Clean and validate inputs
    y_true = np.array(y_true).astype(int).flatten()
    pred1 = np.array(pred1).astype(float).flatten()
    pred2 = np.array(pred2).astype(float).flatten()

    assert len(y_true) == len(pred1) == len(pred2), \
        "Lengths of y_true, pred1, pred2 must match."

    # Sort by decreasing pred1 for DeLong's method
    order = np.argsort(-pred1)
    y = y_true[order]
    preds = np.vstack([pred1[order], pred2[order]])

    label_1_count = int(np.sum(y))

    # Safety checks
    if label_1_count == 0 or label_1_count == len(y):
        raise ValueError("y_true must contain both positive and negative classes.")

    aucs, cov = fast_delong(preds, label_1_count)

    auc1 = aucs[0]
    auc2 = aucs[1]
    diff = auc1 - auc2

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        raise ValueError("Computed variance is non-positive. Check your predictions.")

    z = diff / np.sqrt(var)
    p = 2 * norm.sf(abs(z))

    return auc1, auc2, diff, z, p

```

### Cell 109 — code (execution_count=None)
```python
pairs = [
    ("OVEC", prob_ovec, "OSEC", prob_osec),
    ("OVEC", prob_ovec, "DAZA", prob_daza),
    ("OSEC", prob_osec, "DAZA", prob_daza),
    ("OVEC", prob_ovec, "XGBoost", prob_xgb),
    ("OSEC", prob_osec, "XGBoost", prob_xgb),
    ("DAZA", prob_daza, "XGBoost", prob_xgb),
]

print("\n=== DeLong AUC Comparisons ===")
for n1, p1, n2, p2 in pairs:
    auc1, auc2, diff, z, p = delong_test(y_true, p1, p2)
    print(f"\n{n1} vs {n2}")
    print(f"AUC {n1}: {auc1:.4f}")
    print(f"AUC {n2}: {auc2:.4f}")
    print(f"Difference: {diff:.4f}")
    print(f"z = {z:.4f}")
    print(f"p = {p:.4f}")

```

### Cell 110 — code (execution_count=None)
```python

```

### Cell 111 — code (execution_count=None)
```python
from sklearn.model_selection import RepeatedStratifiedKFold

models_auc_cv = {}
models_f1_cv = {}

for name, maker in model_factories.items():
    aucs, f1s = cross_val_metrics(maker, X_train, y_train, n_splits=5, n_repeats=5)
    models_auc_cv[name] = aucs
    models_f1_cv[name]  = f1s

    print(f"{name} AUCs:", aucs)
    print(f"{name} F1s :", f1s)

```

### Cell 112 — code (execution_count=None)
```python
pairs = [
    ("OVEC", "OSEC"),
    ("OVEC", "1A Stacking (Daza et al., 2024)"),
    ("OSEC", "1A Stacking (Daza et al., 2024)"),
    ("OVEC", "XGBoost"),
    ("OSEC", "XGBoost"),
]

print("\n=== Wilcoxon Tests ===")
for m1, m2 in pairs:
    print(m1, "vs", m2, safe_wilcoxon(models_auc_cv[m1], models_auc_cv[m2]))
```

### Cell 113 — code (execution_count=None)
```python

```

### Cell 114 — code (execution_count=None)
```python

```

### Cell 115 — code (execution_count=None)
```python
"""
FULL EVALUATION PIPELINE — FINAL UNIFIED SCRIPT

Models evaluated:
- OVEC (our ensemble 1)
- OSEC (our proposed XGB-based stacking ensemble)
- 1A Stacking (Daza et al., 2024) - reimplementation
- XGBoost (single model baseline, tuned with GridSearchCV)
- Logistic Regression (LR baseline, tuned with RandomizedSearchCV)

Reviewer coverage:
- Reviewer #1, Concern #10: statistical testing, bootstrapped CIs, DeLong test,
  calibration, decision-curve analysis.
- Reviewer #2, Concern #12: paired Wilcoxon across folds, CIs for AUC and F1,
  claims backed with significance tests.
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, norm
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from xgboost import XGBClassifier


# ============================================================
# 1. USER INPUT / OBJECTS TO DEFINE
# ============================================================
"""
You must define these variables BEFORE running the script:

DATA:
-----
X_train, y_train          : training data for OVEC, OSEC, XGBoost, etc.
X_train_daza, y_train_daza: training data for 1A Stacking (Daza et al., 2024)
X_train_old, y_train_old  : training data used for LR tuning (can be same as X_train,y_train)
X_test, y_test            : test set features and ground-truth labels (binary 0/1)

MODELS (already trained or to be trained here):
-----------------------------------------------
ovec_model : your OVEC ensemble model (must implement predict_proba)
# OSEC will be defined here as calibrated_stacking_xgb (XGB-based stacking)
stacking_classifier : StackingClassifier implementing Daza's 1A stacking
best_bayes, best_ab, best_tree, best_logistic : tuned base learners for OSEC stacking

CROSS-VALIDATION METRICS (per-fold):
------------------------------------
auc_ovec, auc_osec, auc_daza, auc_xgb, auc_lr : lists/arrays of AUC per fold
f1_ovec,  f1_osec,  f1_daza,  f1_xgb,  f1_lr  : lists/arrays of F1 per fold

NOTE:
- auc_osec / f1_osec refer to your OSEC stacking model (same as calibrated_stacking_xgb).
- auc_daza / f1_daza refer to reimplementation of 1A Stacking (Daza et al., 2024).
- auc_xgb / f1_xgb refer to standalone XGBoost (best_xgb below).
- auc_lr / f1_lr refer to Logistic Regression baseline (ys54_logistic_best).
"""


# ============================================================
# 2. TRAIN XGBOOST BASELINE (GridSearchCV)
# ============================================================

print("\n=== Training XGBoost with GridSearchCV ===")

param_grid_xgb = {
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

xgb = XGBClassifier()
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
print("Best Parameters of XGBoost Classifier Grid Search:", best_xgb)

# Fit best model on training data
best_xgb.fit(X_train, y_train)

# Training accuracy
train_accuracy_xgb = best_xgb.score(X_train, y_train)
print(f"Training Accuracy (XGB): {train_accuracy_xgb:.4f}")

# Validation accuracy (CV)
cv_scores_xgb = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_xgb = cv_scores_xgb.mean()
print(f"Validation Accuracy (XGB): {validation_accuracy_xgb:.4f}")

# Test predictions
y_pred_xgb = best_xgb.predict(X_test)

print("\nClassification Report (XGB):")
print(classification_report(y_test, y_pred_xgb))

test_accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Test Accuracy (XGB): {test_accuracy_xgb:.4f}")

# Probabilities for evaluation
y_pred_xgb = best_xgb.predict(X_test)
prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = (prob_xgb >= 0.5).astype(int)

train_accuracy_xgb = best_xgb.score(X_train, y_train)
test_accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

# AUC-ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, prob_xgb)
auc_xgb = auc(fpr, tpr)
```

### Cell 116 — code (execution_count=None)
```python
# ============================================================
# 3. TRAIN/CALIBRATE STACKING MODELS (DAZA + OSEC)
# ============================================================

print("\n=== Training Stacking Models ===")

# 3.1 Reimplementation of 1A Stacking (Daza et al., 2024)
calibrated_stacking_daza = CalibratedClassifierCV(
    stacking_classifier_daza, method='sigmoid', cv=5
)
calibrated_stacking_daza.fit(X_train_daza, y_train_daza)

# 3.2 Our proposed OSEC = XGB-based stacking (using best_xgb as final_estimator)
stacking_classifier_xgb = StackingClassifier(
    cv=5,
    estimators=[
        ('Bayes', best_bayes),
        ('AdaBoost', best_ab),
        ('DecisionTree', best_tree),
        ('LogisticRegression', best_logistic),
    ],
    final_estimator=best_xgb,
)

# Fit stacking_classifier_xgb before passing to CalibratedClassifierCV
stacking_classifier_xgb.fit(X_train, y_train)

calibrated_stacking_xgb = CalibratedClassifierCV(
    stacking_classifier_xgb, method='sigmoid', cv='prefit'
)
calibrated_stacking_xgb.fit(X_train, y_train)


# ============================================================
# 4. TEST-SET PROBABILITIES FOR ALL MODELS
# ============================================================

print("\n=== Generating Test-set Probabilities ===")

# OVEC (your voting/ensemble model)
prob_ovec = voting_model.predict_proba(X_test)[:, 1]

# OSEC (our proposed stacking model = calibrated_stacking_xgb)
prob_osec = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]

# 1A Stacking (Daza et al., 2024)
prob_daza = calibrated_stacking_daza.predict_proba(X_test_daza)[:, 1]

# XGBoost baseline (already defined as prob_xgb)
# Test predictions
y_pred_xgb = best_xgb.predict(X_test)
# Probabilities for evaluation
prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

# Logistic Regression baseline (already defined as prob_lr)
#y_pred_lr = ys54_logistic_best.predict(X_test_old)
#prob_lr = ys54_logistic_best.predict_proba(X_test_old)[:, 1]


MODEL_DAZA_NAME = "1A Stacking (Daza et al., 2024)"

models_proba = {
    "OVEC": prob_ovec,
    "OSEC": prob_osec,
    MODEL_DAZA_NAME: prob_daza,
    "XGBoost": prob_xgb,
    #"Logistic Regression": prob_lr,
}

# Use y_test consistently as ground truth
y_true = np.array(y_test)

```

### Cell 117 — code (execution_count=None)
```python
# Calculate AUC and F1 scores for each model
# AUC-ROC Curve
voting_probabilities = voting_model.predict_proba(X_test_old)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_old, voting_probabilities)
auc_ovec = auc(fpr, tpr)


auc_osec = roc_auc_score(y_test, prob_osec)
auc_daza = roc_auc_score(y_test_daza, prob_daza)
auc_xgb = roc_auc_score(y_test, prob_xgb)
#auc_lr = roc_auc_score(y_test_old, prob_lr)

f1_ovec = f1_score(y_test_old, voting_predictions)
f1_osec = f1_score(y_test, (prob_osec >= 0.5).astype(int))
f1_daza = f1_score(y_test_daza, (prob_daza >= 0.5).astype(int))
f1_xgb = f1_score(y_test, (prob_xgb >= 0.5).astype(int))
#f1_lr = f1_score(y_test_old, (prob_lr >= 0.5).astype(int))
```

### Cell 118 — code (execution_count=None)
```python

```

### Cell 119 — code (execution_count=None)
```python
# 🔧 STEP 1 — Import clone
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

```

### Cell 120 — code (execution_count=None)
```python
# 🔥 STEP 2 — Define a reusable function
#
def cross_val_metrics(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    aucs = []
    f1s = []

    for train_idx, test_idx in kf.split(X):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]

        # Important: clone model fresh for each fold
        model_cv = clone(model)

        # Fit model
        model_cv.fit(X_train_cv, y_train_cv)

        # Predict probabilities (required for ROC AUC)
        y_proba = model_cv.predict_proba(X_test_cv)[:, 1]

        # Store metrics
        aucs.append(roc_auc_score(y_test_cv, y_proba))
        f1s.append(f1_score(y_test_cv, (y_proba >= 0.5).astype(int)))

    return np.array(aucs), np.array(f1s)

```

### Cell 121 — code (execution_count=None)
```python
# STEP 3 — Compute CV Metrics for Each Model
models_auc_cv = {}
models_f1_cv = {}

for name, model in {
    "OVEC": voting_model,
    "OSEC": calibrated_stacking_xgb,
    MODEL_DAZA_NAME: calibrated_stacking_daza,
    "XGBoost": best_xgb,
}.items():

    aucs, f1s = cross_val_metrics(model, X_daza.values, y_daza.values, cv=5)

    models_auc_cv[name] = aucs
    models_f1_cv[name]  = f1s
```

### Cell 122 — code (execution_count=None)
```python
# 🔧 STEP 4 — Run Wilcoxon Correctly
def safe_wilcoxon(a, b):
    a, b = np.array(a), np.array(b)
    if np.all(a == b):
        return "All folds identical — Wilcoxon not defined."
    stat, p = wilcoxon(a, b)
    return stat, p
pairs_for_wilcoxon = [
    ("OVEC", "OSEC"),
    ("OVEC", MODEL_DAZA_NAME),
    ("OSEC", MODEL_DAZA_NAME),
    ("OVEC", "XGBoost"),
    ("OSEC", "XGBoost"),
    (MODEL_DAZA_NAME, "XGBoost"),
]

print("\n=== Paired Wilcoxon Tests on Cross-Validation Folds ===")
for m1, m2 in pairs_for_wilcoxon:
    print(f"\n{m1} vs {m2}")
    print("  AUC Wilcoxon:", safe_wilcoxon(models_auc_cv[m1], models_auc_cv[m2]))
    print("  F1  Wilcoxon:", safe_wilcoxon(models_f1_cv[m1], models_f1_cv[m2]))

```

### Cell 123 — code (execution_count=None)
```python
# These CV metrics must be defined by you (lists/arrays of same length):
models_auc_cv = {
    "OVEC": auc_ovec,
    "OSEC": auc_osec,
    MODEL_DAZA_NAME: auc_daza,
    "XGBoost": auc_xgb,
    #"Logistic Regression": auc_lr,
}

models_f1_cv = {
    "OVEC": f1_ovec,
    "OSEC": f1_osec,
    MODEL_DAZA_NAME: f1_daza,
    "XGBoost": f1_xgb,
    #"Logistic Regression": f1_lr,
}


# ============================================================
# 5. WILCOXON SIGNIFICANCE TESTS ACROSS FOLDS (Reviewer #2)
# ============================================================

def safe_wilcoxon(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.all(a == b):
        return "All folds identical — Wilcoxon not defined."
    stat, p = wilcoxon(a, b)
    return stat, p

pairs_for_wilcoxon = [
    ("OVEC", "OSEC"),
    ("OVEC", MODEL_DAZA_NAME),
    ("OSEC", MODEL_DAZA_NAME),
    ("OVEC", "XGBoost"),
    ("OSEC", "XGBoost"),
    (MODEL_DAZA_NAME, "XGBoost"),
]

print("\n=== Paired Wilcoxon Tests on Cross-Validation Folds ===")
for m1, m2 in pairs_for_wilcoxon:
    print(f"\n{m1} vs {m2}")
    print("  AUC Wilcoxon:", safe_wilcoxon(models_auc_cv[m1], models_auc_cv[m2]))
    print("  F1  Wilcoxon:", safe_wilcoxon(models_f1_cv[m1], models_f1_cv[m2]))


```

### Cell 124 — code (execution_count=None)
```python
for name, prob in models_proba.items():
    print(name, len(prob), len(y_true))

```

### Cell 125 — code (execution_count=None)
```python
for name, prob in models_proba.items():
    if len(prob) != len(y_true):
        print("❌ Mismatch:", name, len(prob), len(y_true))
    else:
        print("✔ OK:", name)

```

### Cell 126 — code (execution_count=None)
```python

# ============================================================
# 6. BOOTSTRAPPED 95% CIs FOR AUC & F1 (Reviewers #1 & #2)
# ============================================================

def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, alpha=0.05, random_state=42):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric(y_true[idx], y_pred[idx]))
    stats = np.array(stats)
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(np.mean(stats)), (float(lower), float(upper))

boot_results = {}

print("\n=== Bootstrapped 95% Confidence Intervals (Test Set) ===")
for name, prob in models_proba.items():
    auc_mean, auc_ci = bootstrap_ci(y_true, prob, roc_auc_score)
    f1_mean, f1_ci = bootstrap_ci(y_true, (prob >= 0.5).astype(int), f1_score)
    boot_results[name] = {
        "auc_mean": auc_mean,
        "auc_ci": auc_ci,
        "f1_mean": f1_mean,
        "f1_ci": f1_ci,
    }
    print(f"\n{name}:")
    print(f"  AUC mean={auc_mean:.4f}, 95% CI={auc_ci}")
    print(f"  F1  mean={f1_mean:.4f}, 95% CI={f1_ci}")


# ============================================================
# 7. DELONG TEST FOR AUC COMPARISON (Reviewer #1)
# ============================================================

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast implementation of DeLong's method for two correlated ROC curves.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    tx = np.apply_along_axis(compute_midrank, 1, pos)
    ty = np.apply_along_axis(compute_midrank, 1, neg)
    tz = np.apply_along_axis(compute_midrank, 1, predictions_sorted_transposed)

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)

    v01 = tx - tx.mean(axis=1, keepdims=True)
    v10 = ty - ty.mean(axis=1, keepdims=True)

    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s

def delong_roc_test(y_true, pred1, pred2):
    """
    DeLong test comparing AUCs of two correlated ROC curves.
    """
    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)

    order = np.argsort(-pred1)
    y = y_true[order]
    preds = np.vstack([pred1[order], pred2[order]])

    label_1_count = np.sum(y)
    aucs, cov = fast_delong(preds, label_1_count)
    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = auc_diff / np.sqrt(var)
    p = 2 * norm.sf(abs(z))
    return aucs, z, p

pairs_for_delong = [
    ("OVEC", "OSEC"),
    ("OVEC", MODEL_DAZA_NAME),
    ("OSEC", MODEL_DAZA_NAME),
    ("OVEC", "XGBoost"),
    ("OSEC", "XGBoost"),
    (MODEL_DAZA_NAME, "XGBoost"),
    ("OVEC", "Logistic Regression"),
    ("OSEC", "Logistic Regression"),
    (MODEL_DAZA_NAME, "Logistic Regression"),
]

print("\n=== DeLong AUC Comparisons (Test Set) ===")
for m1, m2 in pairs_for_delong:
    aucs, z, p = delong_roc_test(y_true, models_proba[m1], models_proba[m2])
    print(f"\n{m1} vs {m2}:")
    print(f"  AUC_{m1} = {aucs[0]:.4f}, AUC_{m2} = {aucs[1]:.4f}")
    print(f"  z = {z:.3f}, p = {p:.4f}")


# ============================================================
# 8. CALIBRATION: Brier Scores & Reliability Curves
# ============================================================

print("\n=== Calibration (Brier Scores) ===")
for name, prob in models_proba.items():
    brier = brier_score_loss(y_true, prob)
    print(f"{name}: Brier score = {brier:.4f}")

plt.figure(figsize=(7, 5))
for name, prob in models_proba.items():
    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration (Reliability) Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 9. DECISION-CURVE ANALYSIS (Clinical Utility)
# ============================================================

def decision_curve(y_true, y_prob, thresholds):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    net_benefits = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefits.append(nb)
    return np.array(net_benefits)

thresholds = np.linspace(0.01, 0.5, 50)
prevalence = np.mean(y_true)
nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
nb_none = np.zeros_like(thresholds)

plt.figure(figsize=(8, 6))
for name, prob in models_proba.items():
    nb = decision_curve(y_true, prob, thresholds)
    plt.plot(thresholds, nb, label=name)

plt.plot(thresholds, nb_all, linestyle='--', label="Treat All")
plt.plot(thresholds, nb_none, linestyle='--', label="Treat None")
plt.xlabel("Risk threshold")
plt.ylabel("Net benefit")
plt.title("Decision-Curve Analysis")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 10. ROC CURVES
# ============================================================

plt.figure(figsize=(7, 6))
for name, prob in models_proba.items():
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc_val = roc_auc_score(y_true, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 11. FOREST PLOTS FOR BOOTSTRAPPED CIs (AUC & F1)
# ============================================================

model_names = list(models_proba.keys())
auc_means  = [boot_results[m]["auc_mean"] for m in model_names]
auc_lows   = [boot_results[m]["auc_ci"][0] for m in model_names]
auc_highs  = [boot_results[m]["auc_ci"][1] for m in model_names]
f1_means   = [boot_results[m]["f1_mean"] for m in model_names]
f1_lows    = [boot_results[m]["f1_ci"][0] for m in model_names]
f1_highs   = [boot_results[m]["f1_ci"][1] for m in model_names]

y_pos = np.arange(len(model_names))

plt.figure(figsize=(10, 5))

# AUC CI plot
plt.subplot(1, 2, 1)
for i, name in enumerate(model_names):
    plt.errorbar(
        auc_means[i],
        y_pos[i],
        xerr=[[auc_means[i] - auc_lows[i]], [auc_highs[i] - auc_means[i]]],
        fmt='o',
    )
plt.yticks(y_pos, model_names)
plt.xlabel("AUC")
plt.title("Bootstrapped 95% CI - AUC")
plt.grid(axis='x', alpha=0.3)

# F1 CI plot
plt.subplot(1, 2, 2)
for i, name in enumerate(model_names):
    plt.errorbar(
        f1_means[i],
        y_pos[i],
        xerr=[[f1_means[i] - f1_lows[i]], [f1_highs[i] - f1_means[i]]],
        fmt='o',
    )
plt.yticks(y_pos, [])
plt.xlabel("F1-score")
plt.title("Bootstrapped 95% CI - F1")
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 12. WILCOXON VISUALIZATION: FOLD-WISE DIFFERENCES
# ============================================================

key_pairs_for_plot = [
    ("OVEC", "OSEC"),
    ("OVEC", MODEL_DAZA_NAME),
    ("OSEC", MODEL_DAZA_NAME),
]

n_folds = len(models_auc_cv["OVEC"])
fold_idx = np.arange(1, n_folds + 1)

plt.figure(figsize=(12, 6))

for idx, (m1, m2) in enumerate(key_pairs_for_plot, start=1):
    auc_diff = np.array(models_auc_cv[m1]) - np.array(models_auc_cv[m2])
    f1_diff  = np.array(models_f1_cv[m1]) - np.array(models_f1_cv[m2])

    plt.subplot(2, 3, idx)
    plt.bar(fold_idx, auc_diff)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f"ΔAUC: {m1} - {m2}")
    plt.xlabel("Fold")
    plt.ylabel("Difference")
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, idx + 3)
    plt.bar(fold_idx, f1_diff)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(f"ΔF1: {m1} - {m2}")
    plt.xlabel("Fold")
    plt.ylabel("Difference")
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```