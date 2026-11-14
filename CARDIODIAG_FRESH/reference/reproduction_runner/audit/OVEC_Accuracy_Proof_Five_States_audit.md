# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\five_states_results\Five States Results\OVEC_Accuracy_Proof_Five_States.ipynb`
- SHA-256: `00ea8506524f30e7114b70a11b9fb008218e179ecc1f994f8d447af582455734`
- Cells: 38
- Saved outputs: 27

## Imports
- `collections`
- `google.colab`
- `imblearn.over_sampling`
- `lime`
- `lime.lime_tabular`
- `loras`
- `matplotlib.pyplot`
- `numpy`
- `pandas`
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
- `sklearn.preprocessing`
- `sklearn.svm`
- `sklearn.tree`
- `warnings`
- `xgboost`

## Method keywords
- CalibratedClassifierCV
- GridSearchCV
- LIME
- LoRAS
- RandomizedSearchCV
- SMOTE
- StackingClassifier
- VotingClassifier
- train_test_split

## Risk flags
- **credential_or_mount**: cells [2]
- **notebook_shell**: cells [24]

## Cell sources

### Cell 1 — markdown (execution_count=None)
```text
# **Import libraries and Mount Google drive**
```

### Cell 2 — code (execution_count=None)
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

### Cell 3 — markdown (execution_count=None)
```text
# **Import Dataset**
```

### Cell 4 — code (execution_count=None)
```python
data = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data
data.head()

```

### Cell 5 — markdown (execution_count=None)
```text
# **Checking for Duplocate Values**
```

### Cell 6 — code (execution_count=None)
```python
import pandas as pd

# Check for duplicates
duplicates = data[data.duplicated()]
print(f"Duplicate rows:\n{duplicates}")

# Remove duplicates
data = data.drop_duplicates()

# Verify that duplicates were removed
print(f"Dataset after removing duplicates:\n{data}")

```

### Cell 7 — markdown (execution_count=None)
```text
# **Convert categorical columns into binary**
```

### Cell 8 — code (execution_count=None)
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

### Cell 9 — code (execution_count=None)
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
corr_matrix = data.corr()

# Extract the correlation with respect to the 'HeartDisease' column
heart_disease_corr = corr_matrix['HeartDisease'].drop('HeartDisease')

# Sort the correlations by values (not absolute values to show direction of correlation)
heart_disease_corr_sorted = heart_disease_corr.sort_values()

# Plot the correlations (horizontal bar plot)
plt.figure(figsize=(10, 6))
sns.barplot(x=heart_disease_corr_sorted.values, y=heart_disease_corr_sorted.index, palette='coolwarm')

# Set x-axis limits between -0.6 and 0.6
plt.xlim(-0.7, 0.7)

# Customize the plot's appearance (grid, title, labels)
plt.title(': Correlation with target attribute', fontsize=14)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add a grid for better readability (similar to the image)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

```

### Cell 10 — code (execution_count=None)
```python
X = data.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data['HeartDisease']
print(data)
print("X",X)
print("y",y)

#Target Variables Values
data.HeartDisease.value_counts()
```

### Cell 11 — markdown (execution_count=None)
```text
# **Detect outliers based on percentiles**
```

### Cell 12 — code (execution_count=None)
```python
import numpy as np
import pandas as pd


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
```

### Cell 13 — code (execution_count=None)
```python
X = data_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data_no_outliers['HeartDisease']


#Target Variables Values
data_no_outliers.HeartDisease.value_counts()
X.shape
```

### Cell 14 — code (execution_count=None)
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Count the occurrences of 'HeartDisease' for both outliers and non-outliers
outliers_heart_disease_counts = data['HeartDisease'].value_counts().sort_index()
non_outliers_heart_disease_counts = data['HeartDisease'].value_counts().sort_index()

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

### Cell 15 — code (execution_count=None)
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

### Cell 16 — code (execution_count=None)
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

### Cell 17 — markdown (execution_count=None)
```text
# **Zscore**
```

### Cell 18 — markdown (execution_count=None)
```text

```

### Cell 19 — code (execution_count=None)
```python
# # Define the Z-score threshold
# z_threshold = 3

# # Identify numerical columns
# numerical_cols = data.select_dtypes(include=[np.number]).columns

# # Function to mark outliers as NaN using Z-score
# def mark_outliers_zscore(df, cols, threshold):
#     df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
#     for col in cols:
#         z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
#         df_copy[col] = np.where(z_scores > threshold, np.nan, df_copy[col])
#     return df_copy  # Return the modified DataFrame

# # Store the modified DataFrame
# data_with_nans = mark_outliers_zscore(data, numerical_cols, z_threshold)

# # Check the result
# print("Data after marking outliers as NaN:")
# print(data_with_nans.isnull().sum())
```

### Cell 20 — code (execution_count=None)
```python
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer

# # Assuming your Outliers dataset after removing outliers
# # The label column is assumed to be named 'HeartDisease'

# # Split data into features and labels
# X_Outliers = data_with_nans.drop('HeartDisease', axis=1)  # Features
# y_Outliers = data_with_nans['HeartDisease']               # Labels

# # Separate into heart disease and normal groups
# df_heart_disease = data_with_nans[data_with_nans['HeartDisease'] == 1].copy()
# df_normal = data_with_nans[data_with_nans['HeartDisease'] == 0].copy()

# # Verify data before imputation
# print("Shape of Heart Disease group before numerical imputation: ", df_heart_disease.shape)
# print("Shape of Normal group before numerical imputation: ", df_normal.shape)

# # Impute Numerical Columns
# numerical_cols = X_Outliers.select_dtypes(include=[np.number]).columns
# median_imputer = SimpleImputer(strategy='median')

# df_heart_disease[numerical_cols] = median_imputer.fit_transform(df_heart_disease[numerical_cols])
# df_normal[numerical_cols] = median_imputer.fit_transform(df_normal[numerical_cols])

# print("Shape of Heart Disease group after numerical imputation: ", df_heart_disease.shape)
# print("Shape of Normal group after numerical imputation: ", df_normal.shape)

# # Combine both groups back together after imputation
# df_imputed = pd.concat([df_heart_disease, df_normal], ignore_index=True)

# print("Final Imputed DataFrame: \n", df_imputed.head())
# print("Missing values after imputation: \n", df_imputed.isnull().sum())
```

### Cell 21 — code (execution_count=None)
```python
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 22 — markdown (execution_count=None)
```text
# **Data Balancing**
```

### Cell 23 — code (execution_count=None)
```python
# ! pip install smote_variants==0.5.1
# ! pip install smote-variants
```

### Cell 24 — code (execution_count=None)
```python
# Install LoRAS package
!pip install loras

# Import necessary libraries
import loras
import numpy as np
import pandas as pd

# Assuming you have the dataset loaded already as features (X) and labels (y)
labels = X # Use your features from the dataset
features =y  # Your target (HeartDisease column)

# Separate the majority and minority classes
X_majority_class = X[y == 1]  # Majority class (HeartDisease = 1)
X_minority_class = X[y == 0]  # Minority class (HeartDisease = 0)

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

### Cell 25 — code (execution_count=None)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Plot before balancing
plt.figure(figsize=(8, 4))
plt.bar(Counter(y).keys(), Counter(y).values(), color=['blue', 'red'])
plt.xticks([0, 1], labels=["Minority (0)", "Majority (1)"])
plt.xlabel("Class Labels")
plt.ylabel("Count")
plt.title("Before Balancing")
plt.show()



```

### Cell 26 — code (execution_count=None)
```python
import matplotlib.pyplot as plt

# Count the number of samples in each class after balancing
unique, counts = np.unique(LoRAS_labels, return_counts=True)

# Create a bar plot
plt.figure(figsize=(6, 4))
bars = plt.bar(['HeartDisease = 1', 'HeartDisease = 0'], counts, color=['blue', 'red'])

# Add counts on top of each bar
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(count),
             ha='center', va='bottom', fontsize=10)

plt.title('Class Distribution After Balancing using LoRAS')
plt.ylabel('Number of Samples')
plt.ylim(0, max(counts) + 50)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

```

### Cell 27 — markdown (execution_count=None)
```text
# **Feature Selection**
```

### Cell 28 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)
# X and y should have shapes (253661, 21) and (253661,), respectively

# Apply PCA to reduce the dimensionality to 2
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca10 = pca.fit_transform(LoRAS_features)
```

### Cell 29 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca10, LoRAS_labels, test_size=0.25, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(LoRAS_features, LoRAS_labels, test_size=0.25, random_state=42)
# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 30 — code (execution_count=None)
```python
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Plot heatmap of the component loadings
# plt.figure(figsize=(12, 6))
# sns.heatmap(loading_df, annot=True, cmap='coolwarm', center=0)
# plt.xlabel('Original Features')
# plt.ylabel('Principal Components')
# plt.title('Visualization of Feature Contributions to Each Principal Component')
# plt.show()
```

### Cell 31 — markdown (execution_count=None)
```text
# **Classification using Voting Model without Hyperparameter Tunning**
```

### Cell 32 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np

# Define the base models
knn = KNeighborsClassifier(metric='manhattan', weights='distance')
rf = RandomForestClassifier()
logistic = LogisticRegression(C=0.0001, solver='liblinear')
svm = SVC(C=1, probability=True)  # Set probability=True for AUC-ROC

# Create a list of models
models = [
    ('knn', knn),
    ('rf', rf),
    ('logistic', logistic),
    ('svm', svm)
]

# Combine models into a Voting Classifier (hard voting for classification)
voting_model = VotingClassifier(
    estimators=[(name, model) for name, model in models],
    voting='soft'  # Change to 'soft' if you want to use probabilities for AUC-ROC
)

# Assume you have the data loaded as X_train, X_test, y_train, and y_test
# Example (replace with your dataset):
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Voting Classifier
voting_model.fit(X_train, y_train)

# Make predictions
voting_predictions = voting_model.predict(X_test)

# Calculate metrics
train_accuracy = voting_model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, voting_predictions)
precision = precision_score(y_test, voting_predictions, average='binary')  # Adjust for multi-class if needed
recall = recall_score(y_test, voting_predictions, average='binary')  # Adjust for multi-class if needed
f1 = f1_score(y_test, voting_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft voting would be needed for using probabilities)
if hasattr(voting_model, "predict_proba"):
    voting_probabilities = voting_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, voting_probabilities)
    auc_score = auc(fpr, tpr)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Voting Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard voting."

# Display all metrics
print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Testing Accuracy: {test_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC: {auc_score}')

```

### Cell 33 — markdown (execution_count=None)
```text
# **Classification with Hyperparameter Tunning**
```

### Cell 34 — code (execution_count=None)
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

# 2. Random Forest
param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [2, None],
    'criterion': ['gini', 'entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train, y_train)
rf_best = rf_rs.best_estimator_

# 3. KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train, y_train)
knn_best = knn_rs.best_estimator_

# 4. SVM
param_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train, y_train)
svm_best = svm_rs.best_estimator_

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

### Cell 35 — code (execution_count=None)
```python
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test, voting_predictions)
total = np.sum(cm)
percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Labels for classes (adjust if needed)
class_names = ['Negative', 'Positive']  # Change if needed

# Plot
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Voting Classifier')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Add raw count + percentage in each cell
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percent = percentages[i, j]
        plt.text(j, i, f'{count}\n({percent:.1f}%)',
                 ha="center", va="center",
                 color="white" if count > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.grid(False)
plt.tight_layout()
plt.show()

```

### Cell 36 — code (execution_count=None)
```python
# Function to evaluate and display metrics for a given model
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    fpr, tpr, auc_score = None, None, None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

    print(f"\n========== {name} Performance ==========")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision         : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall            : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score          : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC           : {auc_score:.4f}" if auc_score is not None else "AUC-ROC           : Not Available")
    print("========================================")

# Evaluate all individual models
evaluate_model(logistic_best, X_train, y_train, X_test, y_test, name="Logistic Regression")
evaluate_model(rf_rs, X_train, y_train, X_test, y_test, name="Random Forest")
evaluate_model(knn_best, X_train, y_train, X_test, y_test, name="K-Nearest Neighbors")
evaluate_model(svm_best, X_train, y_train, X_test, y_test, name="Support Vector Machine")

```

### Cell 37 — code (execution_count=None)
```python
pip install lime

```

### Cell 38 — code (execution_count=None)
```python
import lime
import lime.lime_tabular
import numpy as np

# Define actual feature names
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope', 'HeartDisease'
]

# Create a LIME explainer with actual feature names
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,  # Training data
    feature_names=feature_names,  # Actual feature names
    class_names=['No Disease', 'Disease'],  # Class labels
    mode='classification'
)

# Select a random test instance to explain
idx = np.random.randint(0, X_test.shape[0])
sample_instance = X_test[idx].reshape(1, -1)

# Explain the model's prediction for the chosen instance
exp = explainer.explain_instance(
    sample_instance.flatten(),
    voting_model.predict_proba,  # Your model's prediction function

)

# Display the explanation
exp.show_in_notebook()  # If running in Jupyter Notebook
exp.save_to_file('lime_explanation.html')  # Save to an HTML file

```