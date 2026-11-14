# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\five_states_results\Five States Results\Reviewer1_Five_States_Full_Methodology.ipynb`
- SHA-256: `a5d6d00a59356ca8822ad023c7c77ef2fd058b1f78299e1af7c5a233b6b809b7`
- Cells: 76
- Saved outputs: 71

## Imports
- `collections`
- `google.colab`
- `imblearn.over_sampling`
- `lime`
- `lime.lime_tabular`
- `loras`
- `matplotlib.pyplot`
- `matplotlib_venn`
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
- `sklearn.utils`
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
- XAIH
- bootstrap
- train_test_split

## Risk flags
- **credential_or_mount**: cells [2]
- **notebook_shell**: cells [25]

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

### Cell 9 — markdown (execution_count=None)
```text
# **Correlation with the Target Variable**
```

### Cell 10 — code (execution_count=None)
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

### Cell 11 — code (execution_count=None)
```python
X = data.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data['HeartDisease']
print(data)
print("X",X)
print("y",y)

#Target Variables Values
data.HeartDisease.value_counts()
```

### Cell 12 — markdown (execution_count=None)
```text
# **Detect outliers based on percentiles**
```

### Cell 13 — code (execution_count=None)
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

### Cell 14 — code (execution_count=None)
```python
X = data_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data_no_outliers['HeartDisease']


#Target Variables Values
data_no_outliers.HeartDisease.value_counts()
X.shape
```

### Cell 15 — code (execution_count=None)
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

### Cell 16 — code (execution_count=None)
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

### Cell 17 — code (execution_count=None)
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

### Cell 18 — markdown (execution_count=None)
```text
# **Zscore**
```

### Cell 19 — markdown (execution_count=None)
```text

```

### Cell 20 — code (execution_count=None)
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

### Cell 21 — code (execution_count=None)
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

### Cell 22 — code (execution_count=None)
```python
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 23 — markdown (execution_count=None)
```text
# **Data Balancing**
```

### Cell 24 — code (execution_count=None)
```python
# ! pip install smote_variants==0.5.1
# ! pip install smote-variants
```

### Cell 25 — code (execution_count=None)
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

### Cell 26 — code (execution_count=None)
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

### Cell 27 — code (execution_count=None)
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

### Cell 28 — markdown (execution_count=None)
```text
# **Feature Selection by NXAIH-DRC**
```

### Cell 29 — markdown (execution_count=None)
```text
**Correlation with the Target Variable**
```

### Cell 30 — code (execution_count=None)
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define actual feature names
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# Build DataFrame
LoRAS_df = pd.DataFrame(LoRAS_features, columns=feature_names)
LoRAS_df["HeartDisease"] = LoRAS_labels

# Correlation matrix
corr_matrix = LoRAS_df.corr()

# Correlation of features with HeartDisease
heart_disease_corr = corr_matrix["HeartDisease"].drop("HeartDisease")

# Sort correlations
heart_disease_corr_sorted = heart_disease_corr.sort_values()

# Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x=heart_disease_corr_sorted.values,
    y=heart_disease_corr_sorted.index,
    hue=heart_disease_corr_sorted.index,
    dodge=False,
    palette="coolwarm",
    legend=False
)

# Annotate each bar with correlation value
for i, (value, feature) in enumerate(zip(heart_disease_corr_sorted.values, heart_disease_corr_sorted.index)):
    ax.text(
        x=value + (0.02 if value >= 0 else -0.02),  # offset depending on sign
        y=i,
        s=f"{value:.2f}",
        va="center",
        ha="left" if value >= 0 else "right",
        fontsize=10,
        color="black"
    )

# Set limits
plt.xlim(-0.7, 0.7)

# Labels & title
plt.title("Correlation with HeartDisease", fontsize=14)
plt.xlabel("Correlation Coefficient", fontsize=12)
plt.ylabel("Features", fontsize=12)

# Grid
plt.grid(True, axis="x", linestyle="--", alpha=0.7)

plt.show()

```

### Cell 31 — code (execution_count=None)
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Define actual feature names
# -------------------------------
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# -------------------------------
# 2. Build DataFrame
# -------------------------------
LoRAS_df = pd.DataFrame(LoRAS_features, columns=feature_names)
LoRAS_df["HeartDisease"] = LoRAS_labels

# -------------------------------
# 3. Correlation matrix
# -------------------------------
corr_matrix = LoRAS_df.corr()

# Correlation of features with HeartDisease
heart_disease_corr = corr_matrix["HeartDisease"].drop("HeartDisease")

# -------------------------------
# 4. Select Top-k Features
# -------------------------------
k = 11  # change as needed

# Sort by absolute correlation
top_k_features = heart_disease_corr.abs().sort_values(ascending=False).head(k)

print(f"Top {k} features by correlation with Heart Disease:\n")
print(top_k_features)

# -------------------------------
# 5. Build reduced dataset
# -------------------------------
selected_features = top_k_features.index.tolist()
LoRAS_topk_df = LoRAS_df[selected_features + ["HeartDisease"]]

# -------------------------------
# 6. Plot Top-k Feature Correlations
# -------------------------------
plt.figure(figsize=(8, 6))
sns.barplot(
    x=top_k_features.values,
    y=top_k_features.index,
    palette="viridis"
)
plt.title(f"Top {k} Features by Correlation with Heart Disease")
plt.xlabel("Correlation (absolute)")
plt.ylabel("Features")
plt.show()


```

### Cell 32 — markdown (execution_count=None)
```text
**LIME Importance**
```

### Cell 33 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train_lime, X_test_lime, y_train_lime, y_test_lime = train_test_split(LoRAS_features, LoRAS_labels, test_size=0.25, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(LoRAS_features, LoRAS_labels, test_size=0.25, random_state=42)
# Step 9: Print shapes of the resulting datasets
print(f'Train_lime features shape: {X_train_lime.shape}')
print(f'Test features shape: {X_test_lime.shape}')
print(f'Train_lime labels shape: {y_train_lime.shape}')
print(f'Test labels shape: {y_test_lime.shape}')
```

### Cell 34 — code (execution_count=None)
```python
pip install lime
```

### Cell 35 — code (execution_count=None)
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# -------------------------------
# 1. Load dataset & train a model
# -------------------------------
# Define actual feature names
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

X_train_lime, X_test_lime, y_train_lime, y_test_lime = train_test_split(LoRAS_features, LoRAS_labels, test_size=0.25, random_state=42)

# Define Model
# Combine tuned models into VotingClassifier
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
log_rs.fit(X_train_lime, y_train_lime)
logistic_best = log_rs.best_estimator_

# 2. Random Forest
param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [2, None],
    'criterion': ['gini', 'entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train_lime, y_train_lime)
rf_best = rf_rs.best_estimator_

# 3. KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train_lime, y_train_lime)
knn_best = knn_rs.best_estimator_

# 4. SVM
param_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train_lime, y_train_lime)
svm_best = svm_rs.best_estimator_


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
voting_model.fit(X_train_lime, y_train_lime)

```

### Cell 36 — code (execution_count=None)
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# ---------------------------
# 1. Setup LIME Explainer
# ---------------------------
explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=['No Disease', 'Disease'],
    mode="classification"
)

# ---------------------------
# 2. Build Feature Importance Matrix (all instances)
# ---------------------------
n_samples = X_train_lime.shape[0]
n_features = len(feature_names)

importance_matrix = np.zeros((n_samples, n_features))

for i in range(n_samples):
    exp = explainer.explain_instance(
        data_row=X_train_lime[i],
        predict_fn=voting_model.predict_proba,
        num_features=n_features
    )

    # Parse feature importance
    for feature, weight in exp.as_list():
        for j, fname in enumerate(feature_names):
            if fname in feature:  # substring match (e.g., "Age > 55" contains "Age")
                importance_matrix[i, j] = weight

# Convert to DataFrame for analysis
importance_df = pd.DataFrame(importance_matrix, columns=feature_names)

# ---------------------------
# 3. Correlation Matrix
# ---------------------------
corr_matrix = importance_df.corr()

# ---------------------------
# 4. Global Feature Ranking
# ---------------------------
global_importance = importance_df.abs().mean().sort_values(ascending=False)

K = 11  # number of top features to select
top_k_features = global_importance.head(K)

print("\nTop-K Features (descending order of importance):")
print(top_k_features)
```

### Cell 37 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Get global feature importance from LIME
# ---------------------------
global_importance = importance_df.abs().mean().sort_values()

# ---------------------------
# 2. Plot
# ---------------------------
plt.figure(figsize=(10,6))
colors = global_importance.map(lambda x: 'indianred')  # all positive (abs values)

bars = plt.barh(global_importance.index, global_importance.values, color=colors)

# Annotate values
for bar, value in zip(bars, global_importance.values):
    plt.text(value, bar.get_y() + bar.get_height()/2,
             f"{value:.2f}",
             va='center',
             ha='left')

plt.title("Global Feature Importance (LIME)")
plt.xlabel("Mean Absolute Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

```

### Cell 38 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import pandas as pd

# Example: assume we already have
# global_importance (Series) -> feature importance scores
# heart_disease_corr (Series) -> correlation with target

k = 11

# Top-k by importance
top_importance = global_importance.head(k)

# Top-k by correlation
top_corr = heart_disease_corr.abs().sort_values(ascending=False).head(k)

# Merge results
df_compare = pd.DataFrame({
    "Importance": top_importance,
    "Correlation": top_corr
}).fillna(0)  # fill NaN if feature missing in one list

# Plot
df_compare.plot(kind="bar", figsize=(12,6))
plt.title("Comparison of Feature Importance vs Correlation with Target")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.show()

```

### Cell 39 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import pandas as pd

# Example: assume we already have
# global_importance (Series) -> feature importance scores (already sorted)
# heart_disease_corr (Series) -> correlation with target

k = 11

# Top-k by importance
top_importance = global_importance.head(k)

# Top-k by correlation
top_corr = heart_disease_corr.abs().sort_values(ascending=False).head(k)

# Combine both (union of features from both lists)
features_union = list(set(top_importance.index).union(set(top_corr.index)))

# Build dataframe
df_compare = pd.DataFrame({
    "Importance": global_importance.reindex(features_union),
    "Correlation": heart_disease_corr.abs().reindex(features_union)
}).fillna(0)

# Sort by importance for nicer plot
df_compare = df_compare.sort_values("Importance", ascending=True)

# Plot horizontal grouped bar chart
ax = df_compare.plot(
    kind="barh",
    figsize=(10, 7),
    width=0.8
)

plt.title("Feature Importance vs Correlation with Heart Disease")
plt.xlabel("Score")
plt.ylabel("Features")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

```

### Cell 40 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import pandas as pd

# Assume you already have:
# global_importance (Series) -> feature importance scores (sorted)
# heart_disease_corr (Series) -> correlation with target

k = 11

# Top-k by importance
top_importance = global_importance.head(k)

# Top-k by correlation
top_corr = heart_disease_corr.abs().sort_values(ascending=False).head(k)

# Find overlap
overlap_features = set(top_importance.index).intersection(set(top_corr.index))

# Union of features (all that appear in either list)
features_union = list(set(top_importance.index).union(set(top_corr.index)))

# Build dataframe
df_compare = pd.DataFrame({
    "Importance": global_importance.reindex(features_union),
    "Correlation": heart_disease_corr.abs().reindex(features_union)
}).fillna(0)

# Sort by importance
df_compare = df_compare.sort_values("Importance", ascending=True)

# Plot horizontal bars
fig, ax = plt.subplots(figsize=(10, 7))
df_compare.plot(kind="barh", ax=ax, width=0.8)

# Highlight overlapping features with a star (*)
for i, feature in enumerate(df_compare.index):
    if feature in overlap_features:
        ax.text(
            df_compare["Importance"].max()*1.02,   # place slightly outside the bar
            i,
            "★",
            color="red", fontsize=14, va="center"
        )

plt.title("Feature Importance vs Correlation with Heart Disease")
plt.xlabel("Score")
plt.ylabel("Features")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

```

### Cell 41 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Assume you already have:
# top_importance (Series) -> top-k features by model importance
# top_corr (Series) -> top-k features by correlation

# Convert to sets
importance_set = set(top_importance.index)
correlation_set = set(top_corr.index)

# Create Venn diagram
plt.figure(figsize=(6,6))
venn = venn2(
    [importance_set, correlation_set],
    set_labels=("Top-K Importance", "Top-K Correlation")
)

plt.title("Overlap of Top-K Features: Importance vs Correlation")
plt.show()

```

### Cell 42 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Assume you already have:
# top_importance (Series) -> top-k features by model importance
# top_corr (Series) -> top-k features by correlation

# Convert to sets
importance_set = set(top_importance.index)
correlation_set = set(top_corr.index)

# Create Venn diagram
plt.figure(figsize=(6,6))
venn = venn2(
    [importance_set, correlation_set],
    set_labels=("Top-K Importance", "Top-K Correlation")
)
plt.title("Overlap of Top-K Features: Importance vs Correlation")
plt.show()

# Print details
print("🔹 Features in both:", importance_set & correlation_set)
print("🔸 Features only in Importance:", importance_set - correlation_set)
print("🔸 Features only in Correlation:", correlation_set - importance_set)

```

### Cell 43 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Assume you already have:
# top_importance (Series) -> top-k features by model importance
# top_corr (Series) -> top-k features by correlation

# Convert to sets
importance_set = set(top_importance.index)
correlation_set = set(top_corr.index)

# Create Venn diagram
plt.figure(figsize=(6,6))
venn = venn2(
    [importance_set, correlation_set],
    set_labels=("Top-K Importance", "Top-K Correlation")
)
plt.title("Overlap of Top-K Features: Importance vs Correlation")
plt.show()

# Overlap sets
overlap = importance_set & correlation_set
importance_only = importance_set - correlation_set
correlation_only = correlation_set - importance_set

# Print details
print("🔹 Features in both:", overlap)
print("🔸 Features only in Importance:", importance_only)
print("🔸 Features only in Correlation:", correlation_only)

# Quantify overlap
union_size = len(importance_set | correlation_set)
overlap_size = len(overlap)
overlap_percent = (overlap_size / union_size) * 100 if union_size > 0 else 0

print(f"\n✅ Overlap: {overlap_size}/{union_size} features "
      f"({overlap_percent:.2f}% agreement)")

```

### Cell 44 — code (execution_count=None)
```python
import matplotlib.pyplot as plt

# Assume you already have:
# global_importance (Series) -> feature importance scores (sorted)
# heart_disease_corr (Series) -> correlation with target

# Function to compute overlap for different k
def compute_overlap(global_importance, heart_disease_corr, k_values):
    results = []
    for k in k_values:
        top_importance = set(global_importance.head(k).index)
        top_corr = set(heart_disease_corr.abs().sort_values(ascending=False).head(k).index)

        overlap = top_importance & top_corr
        union = top_importance | top_corr

        overlap_percent = (len(overlap) / len(union)) * 100 if len(union) > 0 else 0
        results.append((k, len(overlap), len(union), overlap_percent))
    return results

# Choose k values to test
k_values = [3, 5, 7, 9, 11]

# Run overlap computation
overlap_results = compute_overlap(global_importance, heart_disease_corr, k_values)

# Print results
for k, overlap_size, union_size, overlap_percent in overlap_results:
    print(f"k={k}: Overlap = {overlap_size}/{union_size} ({overlap_percent:.2f}%)")

# Plot overlap percentage vs k
ks = [r[0] for r in overlap_results]
percentages = [r[3] for r in overlap_results]

plt.figure(figsize=(8,5))
plt.plot(ks, percentages, marker='o', linestyle='-', color='b')
plt.title("Overlap Percentage Between Top-K Features (Importance vs Correlation)")
plt.xlabel("k (Top-K Features)")
plt.ylabel("Overlap Percentage (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

```

### Cell 45 — code (execution_count=None)
```python
import matplotlib.pyplot as plt

# Assume you already have:
# global_importance (Series) -> feature importance scores (sorted)
# heart_disease_corr (Series) -> correlation with target

# Function to compute overlap for different k
def compute_overlap(global_importance, heart_disease_corr, k_values):
    results = []
    for k in k_values:
        top_importance = set(global_importance.head(k).index)
        top_corr = set(heart_disease_corr.abs().sort_values(ascending=False).head(k).index)

        overlap = top_importance & top_corr
        union = top_importance | top_corr

        overlap_percent = (len(overlap) / len(union)) * 100 if len(union) > 0 else 0
        results.append((k, len(overlap), len(union), overlap_percent))
    return results

# Choose k values to test
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Run overlap computation
overlap_results = compute_overlap(global_importance, heart_disease_corr, k_values)

# Print results
for k, overlap_size, union_size, overlap_percent in overlap_results:
    print(f"k={k}: Overlap = {overlap_size}/{union_size} ({overlap_percent:.2f}%)")

# Plot overlap percentage vs k
ks = [r[0] for r in overlap_results]
percentages = [r[3] for r in overlap_results]

plt.figure(figsize=(8,5))
plt.plot(ks, percentages, marker='o', linestyle='-', color='b')
plt.title("Overlap Percentage Between Top-K Features (Importance vs Correlation)")
plt.xlabel("k (Top-K Features)")
plt.ylabel("Overlap Percentage (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

```

### Cell 46 — code (execution_count=None)
```python
# Assume you already have:
# global_importance (Series) -> feature importance scores (sorted)
# heart_disease_corr (Series) -> correlation with target

def overlap_features(global_importance, heart_disease_corr, k_values):
    overlap_dict = {}

    for k in k_values:
        top_importance = set(global_importance.head(k).index)
        top_corr = set(heart_disease_corr.abs().sort_values(ascending=False).head(k).index)

        overlap = top_importance & top_corr
        overlap_dict[k] = list(overlap)  # store overlapping feature names

    return overlap_dict

# Example usage
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
overlap_dict = overlap_features(global_importance, heart_disease_corr, k_values)

# Print results
for k, features in overlap_dict.items():
    print(f"k={k}: Common Features ({len(features)}): {features}")

```

### Cell 47 — code (execution_count=None)
```python
def overlap_by_threshold(global_importance, heart_disease_corr, T_importance, T_corr):
    """
    Compare overlap of features using thresholds instead of top-k selection.
    """
    # Select features above thresholds
    selected_importance = set(global_importance[global_importance >= T_importance].index)
    selected_corr = set(heart_disease_corr.abs()[heart_disease_corr.abs() >= T_corr].index)

    # Overlap and differences
    overlap = selected_importance & selected_corr
    importance_only = selected_importance - selected_corr
    corr_only = selected_corr - selected_importance

    return {
        "importance_selected": selected_importance,
        "correlation_selected": selected_corr,
        "overlap": overlap,
        "importance_only": importance_only,
        "corr_only": corr_only
    }

# Example usage
T_importance = 0.05   # threshold for feature importance (adjust as per scale)
T_corr = 0.2          # threshold for correlation

results = overlap_by_threshold(global_importance, heart_disease_corr, T_importance, T_corr)

print("✅ Features above importance threshold:", results["importance_selected"])
print("✅ Features above correlation threshold:", results["correlation_selected"])
print("🔹 Common Features:", results["overlap"])
print("🔸 Only in Importance:", results["importance_only"])
print("🔸 Only in Correlation:", results["corr_only"])

```

### Cell 48 — code (execution_count=None)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def overlap_threshold_heatmap(global_importance, heart_disease_corr,
                              T_importance_values, T_corr_values):
    results = []

    for T_imp in T_importance_values:
        for T_corr in T_corr_values:
            # Select features above thresholds
            selected_importance = set(global_importance[global_importance >= T_imp].index)
            selected_corr = set(heart_disease_corr.abs()[heart_disease_corr.abs() >= T_corr].index)

            # Compute overlap
            overlap = selected_importance & selected_corr
            union = selected_importance | selected_corr
            overlap_percent = (len(overlap) / len(union) * 100) if union else 0

            results.append((T_imp, T_corr, overlap_percent, len(overlap), len(union)))

    # Convert to DataFrame
    df = pd.DataFrame(results, columns=["T_importance", "T_corr", "Overlap_%", "Overlap", "Union"])

    # Pivot for heatmap
    pivot = df.pivot(index="T_importance", columns="T_corr", values="Overlap_%")

    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "Overlap %"})
    plt.title("Overlap % of Features by Thresholds")
    plt.xlabel("Correlation Threshold (T_corr)")
    plt.ylabel("Importance Threshold (T_importance)")
    plt.show()

    return df

# Example usage
T_importance_values = np.linspace(0.01, 0.1, 5)  # adjust depending on scale of importance
T_corr_values = np.linspace(0.1, 0.5, 5)         # adjust depending on correlation

df_results = overlap_threshold_heatmap(global_importance, heart_disease_corr,
                                       T_importance_values, T_corr_values)

print(df_results.head())

```

### Cell 49 — code (execution_count=None)
```python
print(corr_matrix.columns)
```

### Cell 50 — code (execution_count=None)
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def overlap_threshold_heatmap_with_features(global_importance, corr_matrix,
                                            T_importance_values, T_corr_values,
                                            target_col="HeartDisease", decimals=2):
    results = []
    common_features_dict = {}

    for T_imp in T_importance_values:
        for T_corr in T_corr_values:
            # Features above importance threshold (LIME/global importance)
            selected_importance = set(global_importance[global_importance >= T_imp].index)

            # Features above correlation threshold with target
            corr_with_target = corr_matrix[target_col].abs()
            selected_corr = set(corr_with_target[corr_with_target >= T_corr].index)

            # Find overlap
            common_features = selected_importance & selected_corr
            overlap_percent = (len(common_features) / max(1, len(selected_importance | selected_corr))) * 100

            # Save results
            results.append({
                "T_importance": round(T_imp, decimals),
                "T_corr": round(T_corr, decimals),
                "Overlap_%": overlap_percent
            })

            # Store actual features with rounded key
            key = (round(T_imp, decimals), round(T_corr, decimals))
            common_features_dict[key] = list(common_features)

    # Convert to DataFrame for heatmap
    df = pd.DataFrame(results)
    pivot = df.pivot(index="T_importance", columns="T_corr", values="Overlap_%")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Feature Overlap (%) between LIME and Correlation Thresholds")
    plt.ylabel("T_importance (LIME)")
    plt.xlabel("T_corr (Correlation)")
    plt.show()

    return df, common_features_dict


# -------------------------
# Example usage
# -------------------------

# Compute correlation matrix on your dataset
corr_matrix = LoRAS_df.corr()

# Threshold ranges
T_importance_values = np.linspace(0.01, 0.05, 5)   # Example: vary LIME threshold
T_corr_values = np.linspace(0.1, 0.5, 5)           # Example: vary correlation threshold

# Run function
df_results, common_features_dict = overlap_threshold_heatmap_with_features(
    global_importance, corr_matrix,
    T_importance_values, T_corr_values,
    target_col="HeartDisease", decimals=2
)

# ✅ Now querying works without KeyError
print("Common features at T_importance=0.03, T_corr=0.3:")
print(common_features_dict.get((0.03, 0.3), []))

```

### Cell 51 — code (execution_count=None)
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function
def overlap_threshold_heatmap_with_features(global_importance, corr_matrix,
                                            T_importance_values, T_corr_values, target_col):
    results = []
    common_features_dict = {}

    for T_imp in T_importance_values:
        for T_corr in T_corr_values:
            # Features above importance threshold (LIME/global importance)
            selected_importance = set(global_importance[global_importance >= T_imp].index)

            # Features above correlation threshold with target
            corr_with_target = corr_matrix[target_col].abs()
            selected_corr = set(corr_with_target[corr_with_target >= T_corr].index)

            # Find overlap
            common_features = selected_importance & selected_corr
            overlap_percent = (len(common_features) / max(1, len(selected_importance | selected_corr))) * 100

            results.append({
                "T_importance": T_imp,
                "T_corr": T_corr,
                "Overlap_%": overlap_percent
            })

            # Save common features for this threshold pair
            common_features_dict[(T_imp, T_corr)] = list(common_features)

    # Convert overlap results into DataFrame
    df = pd.DataFrame(results)
    pivot = df.pivot(index="T_importance", columns="T_corr", values="Overlap_%")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Feature Overlap (%) between LIME and Correlation Thresholds")
    plt.ylabel("T_importance (LIME)")
    plt.xlabel("T_corr (Correlation)")
    plt.show()

    # Convert common_features_dict into DataFrame for easy export
    df_features = pd.DataFrame.from_dict(
        common_features_dict, orient="index"
    ).reset_index()
    df_features[["T_importance", "T_corr"]] = pd.DataFrame(df_features["index"].tolist(), index=df_features.index)
    df_features = df_features.drop(columns=["index"])

    return df, df_features


# -----------------------------
# Example Usage
# -----------------------------
# corr_matrix = df.corr()   # compute correlation on your dataset
# global_importance = some_importance_series  # should be a Series with feature importance

T_importance_values = np.linspace(0.01, 0.05, 5)   # adjust for your importance scale
T_corr_values = np.linspace(0.1, 0.5, 5)           # adjust for correlation scale

df_results, df_features = overlap_threshold_heatmap_with_features(
    global_importance,
    corr_matrix,
    T_importance_values,
    T_corr_values,
    target_col="HeartDisease"
)

print("\n--- Overlap Percentages ---")
print(df_results.head())

print("\n--- Common Features at Thresholds ---")
print(df_features.head())

```

### Cell 52 — code (execution_count=None)
```python
from sklearn.decomposition import PCA

# -------------------------------
# 1. Set top-k = 11 thresholds
# -------------------------------
k = 11

tau_corr = heart_disease_corr.abs().sort_values(ascending=False).iloc[k-1]
tau_lime = global_importance.sort_values(ascending=False).iloc[k-1]

print("tau_corr (top-11):", tau_corr)
print("tau_lime (top-11):", tau_lime)

# -------------------------------
# 2. Apply NXAIH-DRC Selection
# -------------------------------
selected_features = []   # features kept as-is
pca_features = []        # features for PCA

for feature in feature_names:
    corr_val = abs(heart_disease_corr[feature])
    lime_val = global_importance[feature]

    if corr_val >= tau_corr and lime_val >= tau_lime:
        selected_features.append(feature)   # keep as-is
    elif (corr_val >= tau_corr) ^ (lime_val >= tau_lime):  # XOR
        pca_features.append(feature)       # candidate for PCA

print("Selected (kept as-is):", selected_features)
print("PCA features:", pca_features)

# -------------------------------
# 3. PCA Transformation (fixed n_components=9)
# -------------------------------
X_final = pd.DataFrame()

# Keep features as-is
if selected_features:
    X_final[selected_features] = LoRAS_df[selected_features]

# Apply PCA to disagreement features
if pca_features:
    X_pca = LoRAS_df[pca_features]

    # Fit PCA with 9 components
    pca = PCA(n_components=9)
    X_pca_transformed = pca.fit_transform(X_pca)

    # Create PCA component names
    pca_columns = [f"PCA_{i+1}" for i in range(X_pca_transformed.shape[1])]
    X_pca_df = pd.DataFrame(X_pca_transformed, columns=pca_columns, index=LoRAS_df.index)

    # Concatenate back
    X_final = pd.concat([X_final, X_pca_df], axis=1)

# Add target label
X_final["HeartDisease"] = LoRAS_df["HeartDisease"]

print("\nFinal NXAIH-DRC Dataset Shape:", X_final.shape)
print(X_final.head())

```

### Cell 53 — code (execution_count=None)
```python
from sklearn.decomposition import PCA

# -------------------------------
# 1. Set top-k = 11 thresholds
# -------------------------------
k = 11
for k in range(len(feature_names), len(feature_names)//2, -1):
    # --- Step 1: thresholds from top-k ---
    tau_corr = heart_disease_corr.abs().sort_values(ascending=False).iloc[k-1]
    tau_lime = global_importance.sort_values(ascending=False).iloc[k-1]

    # --- Step 2: Apply rule ---
    if tau_corr != tau_lime:  # thresholds not same (XOR) → PCA on all features
        pca = PCA(n_components=9, random_state=42)
        final_df = pd.DataFrame(
            pca.fit_transform(LoRAS_df[feature_names]),
            columns=[f"PCA_{i+1}" for i in range(9)]
        )
        method = "PCA applied on all features"
        kept_features = []
    else:  # thresholds same → keep features that satisfy both thresholds
        selected_features = []
        for feature in feature_names:
            corr_val = abs(heart_disease_corr[feature])
            lime_val = global_importance[feature]
            if corr_val > tau_corr and lime_val > tau_lime:
                selected_features.append(feature)
        final_df = LoRAS_df[selected_features].copy()
        method = "As-is (kept features)"
        kept_features = selected_features

    # --- Step 3: Report ---
    print(f"\n=== Top-{k} Iteration ===")
    print("Thresholds -> τ_corr:", tau_corr, ", τ_LIME:", tau_lime)
    print("Selection Method:", method)
    print("Final Dataset Shape:", final_df.shape)
    if kept_features:
        print("Kept Features:", kept_features)
```

### Cell 54 — code (execution_count=None)
```python

```

### Cell 55 — code (execution_count=None)
```python
from sklearn.decomposition import PCA

target_col = "HeartDisease"   # adjust if target has a different name
X = LoRAS_df.drop(columns=[target_col])
y = LoRAS_df[target_col]
k=11

for k in range(len(feature_names), len(feature_names)//2, -1):
    # thresholds from top-k features
    tau_corr = heart_disease_corr.abs().sort_values(ascending=False).iloc[k-1]
    tau_lime = global_importance.sort_values(ascending=False).iloc[k-1]

    print(f"\n=== Top-{k} Features in Descending Order ===")
    print(f"Thresholds -> τ_corr: {tau_corr} , τ_LIME: {tau_lime}")
```

### Cell 56 — code (execution_count=None)
```python
from sklearn.decomposition import PCA

target_col = "HeartDisease"
X = LoRAS_df.drop(columns=[target_col])
y = LoRAS_df[target_col]

all_features = set(feature_names)   # ✅ fix

for k in range(len(feature_names), len(feature_names)//2, -1):
    # thresholds from top-k features
    tau_corr = heart_disease_corr.abs().sort_values(ascending=False).iloc[k-1]
    tau_lime = global_importance.sort_values(ascending=False).iloc[k-1]

    print(f"\n=== Top-{k} Features in Descending Order ===")
    print(f"Thresholds -> τ_corr: {tau_corr} , τ_LIME: {tau_lime}")

    # Sets of selected features
    selected_corr = set(heart_disease_corr[heart_disease_corr.abs() >= tau_corr].index)
    selected_lime = set(global_importance[global_importance >= tau_lime].index)

    # Apply Equation 46
    overlap = selected_corr & selected_lime              # AND
    xor_features = selected_corr ^ selected_lime         # XOR
    neither = all_features - (selected_corr | selected_lime)  # neither condition

    if len(xor_features) > 0:  # XOR → PCA on all features
        pca = PCA(n_components=9, random_state=42)
        pca_components = pca.fit_transform(X)
        pca_feature_names = [f"PCA_{i+1}" for i in range(9)]
        final_df = pd.DataFrame(pca_components, columns=pca_feature_names)
        final_df[target_col] = y.values
        print("Selection Method: PCA applied on all features")
    elif len(overlap) > 0:  # AND → keep as-is
        final_df = X[list(overlap)].copy()
        final_df[target_col] = y.values
        print("Selection Method: As-is (kept features)")
        print("Kept Features:", list(overlap))
    else:  # neither → features removed
        final_df = pd.DataFrame({target_col: y.values})
        print("Selection Method: No features retained")

    print("Final Dataset Shape:", final_df.shape)
```

### Cell 57 — code (execution_count=None)
```python
from sklearn.decomposition import PCA

target_col = "HeartDisease"   # adjust if target column differs
X = LoRAS_df.drop(columns=[target_col])
y = LoRAS_df[target_col]

for k in range(len(feature_names), len(feature_names)//2, -1):
    tau_corr = heart_disease_corr.abs().sort_values(ascending=False).iloc[k-1]
    tau_lime = global_importance.sort_values(ascending=False).iloc[k-1]

    print(f"\n=== Top-{k} Features in Descending Order ===")
    print(f"Thresholds -> τ_corr: {tau_corr} , τ_LIME: {tau_lime}")

    if tau_corr != tau_lime:   # XOR → PCA
        pca = PCA(n_components=9)
        X_pca = pca.fit_transform(X)
        final_df = pd.DataFrame(X_pca, index=X.index)
        final_df[target_col] = y.values
        print("Selection Method: PCA applied on all features")
    else:                      # AND → Keep original features
        kept_features = X.columns[:k]
        final_df = X[kept_features].copy()
        final_df[target_col] = y.values
        print("Selection Method: As-is (kept features)")
        print("Kept Features:", list(kept_features))

    print("Final Dataset Shape:", final_df.shape)
```

### Cell 58 — markdown (execution_count=None)
```text
**Feature Selection**
```

### Cell 59 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)
# X and y should have shapes (253661, 21) and (253661,), respectively

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(LoRAS_features)
```

### Cell 60 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, LoRAS_labels, test_size=0.25, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(LoRAS_features, LoRAS_labels, test_size=0.25, random_state=42)
# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 61 — code (execution_count=None)
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

### Cell 62 — markdown (execution_count=None)
```text
# **Classification using Voting Model without Hyperparameter Tunning**
```

### Cell 63 — code (execution_count=None)
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

### Cell 64 — markdown (execution_count=None)
```text
# **Classification with Hyperparameter Tunning**
```

### Cell 65 — code (execution_count=None)
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

### Cell 66 — code (execution_count=None)
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

### Cell 67 — code (execution_count=None)
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

### Cell 68 — code (execution_count=None)
```python
pip install lime

```

### Cell 69 — code (execution_count=None)
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

### Cell 70 — markdown (execution_count=None)
```text
# **Confidence Intervals**
```

### Cell 71 — code (execution_count=None)
```python
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
```

### Cell 72 — code (execution_count=None)
```python
# Refit tuned base models on 9-feature training data
logistic_best.fit(X_train, y_train)
rf_best.fit(X_train, y_train)
knn_best.fit(X_train, y_train)
svm_best.fit(X_train, y_train)

# Voting Classifier (all on 9 features now)
voting_model = VotingClassifier(
    estimators=[
        ('knn', knn_best),
        ('rf', rf_best),
        ('logistic', logistic_best),
        ('svm', svm_best)
    ],
    voting='soft',
    weights=[1, 2, 1, 2]
)

voting_model.fit(X_train, y_train)

```

### Cell 73 — code (execution_count=None)
```python

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def bootstrap_ci(model, X_test, y_test, n_bootstrap=1000, alpha=0.95):
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": []
    }

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    n_samples = len(y_test)

    for _ in range(n_bootstrap):
        # Sample indices with replacement
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        X_res, y_res = X_test[indices], y_test[indices]

        # Predictions
        y_pred = model.predict(X_res)
        y_prob = model.predict_proba(X_res)[:, 1]

        # Collect metrics
        metrics["accuracy"].append(accuracy_score(y_res, y_pred))
        metrics["precision"].append(precision_score(y_res, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_res, y_pred))
        metrics["f1"].append(f1_score(y_res, y_pred))
        fpr, tpr, _ = roc_curve(y_res, y_prob)
        metrics["auc"].append(auc(fpr, tpr))

    # Compute mean + CI bounds
    ci_results = {}
    lower = (1 - alpha) / 2
    upper = 1 - lower

    for metric, values in metrics.items():
        ci_results[metric] = (
            np.mean(values),                              # Mean
            np.percentile(values, lower * 100),           # Lower bound
            np.percentile(values, upper * 100)            # Upper bound
        )

    return ci_results


# 🔹 Apply bootstrapped CI on Voting Classifier
ci_results = bootstrap_ci(voting_model, X_test, y_test, n_bootstrap=1000)

# 🔹 Display results
print("\n========== Model Performance with 95% Confidence Intervals ==========")
for metric, (mean_val, lower, upper) in ci_results.items():
    print(f"{metric.capitalize():<10}: {mean_val:.4f} "
          f"[{lower:.4f}, {upper:.4f}]")
print("====================================================================")


```

### Cell 74 — markdown (execution_count=None)
```text
# **Calibration Metrics**
```

### Cell 75 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# -----------------------------
# 1. Get model predictions
# -----------------------------
y_probs = voting_model.predict_proba(X_test)[:, 1]

# -----------------------------
# 2. Calibration Curve
# -----------------------------
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', label="Voting Classifier")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve (Reliability Plot)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 3. Brier Score
# -----------------------------
brier = brier_score_loss(y_test, y_probs)
print("Brier Score:", round(brier, 4))

# -----------------------------
# 4. Decision Curve Analysis
# -----------------------------
def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y_true)
    net_benefit = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        # Net benefit formula
        nb = (tp/n) - (fp/n) * (t/(1-t))
        net_benefit.append(nb)

    return thresholds, net_benefit

thresholds, net_benefit = decision_curve_analysis(y_test, y_probs)

plt.figure(figsize=(6,5))
plt.plot(thresholds, net_benefit, label="Voting Classifier")
plt.plot(thresholds, [0]*len(thresholds), linestyle="--", label="Treat None")
plt.plot(thresholds, [np.mean(y_test)]*len(thresholds), linestyle="--", label="Treat All")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA)")
plt.legend()
plt.grid(True)
plt.show()

```

### Cell 76 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

# ==============================
# 1. Define helper functions
# ==============================
def bootstrap_ci_metric(y_true, y_pred, y_probs, metric_func, n_bootstrap=1000, ci=95):
    """Compute metric with bootstrap CI."""
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        if metric_func.__name__ == "roc_auc_score":
            scores.append(metric_func(y_true[idx], y_probs[idx]))
        else:
            scores.append(metric_func(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (100-ci)/2)
    upper = np.percentile(scores, 100-(100-ci)/2)
    return np.mean(scores), (lower, upper)

def bootstrap_brier(y_true, y_probs, n_bootstrap=1000, ci=95):
    """Compute Brier score with bootstrap CI."""
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        scores.append(brier_score_loss(y_true[idx], y_probs[idx]))
    lower = np.percentile(scores, (100-ci)/2)
    upper = np.percentile(scores, 100-(100-ci)/2)
    return np.mean(scores), (lower, upper)

def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y_true)
    net_benefit = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp/n) - (fp/n) * (t/(1-t))
        net_benefit.append(nb)
    return np.array(thresholds), np.array(net_benefit)

def bootstrap_dca(y_true, y_prob, n_bootstrap=500, thresholds=np.linspace(0.01, 0.99, 50)):
    all_net_benefits = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        _, nb = decision_curve_analysis(y_true[idx], y_prob[idx], thresholds)
        all_net_benefits.append(nb)
    all_net_benefits = np.array(all_net_benefits)
    mean_nb = np.mean(all_net_benefits, axis=0)
    lower = np.percentile(all_net_benefits, 2.5, axis=0)
    upper = np.percentile(all_net_benefits, 97.5, axis=0)
    return mean_nb, lower, upper

# ==============================
# 2. Evaluate all models
# ==============================
models = {
    "Voting Classifier": voting_model,
    "Random Forest": rf_best,
    "SVM": svm_best,
    "KNN": knn_best,
    "Logistic Regression": logistic_best
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    acc, acc_ci = bootstrap_ci_metric(y_test, y_pred, y_probs, accuracy_score)
    prec, prec_ci = bootstrap_ci_metric(y_test, y_pred, y_probs, precision_score)
    rec, rec_ci = bootstrap_ci_metric(y_test, y_pred, y_probs, recall_score)
    f1, f1_ci = bootstrap_ci_metric(y_test, y_pred, y_probs, f1_score)
    auc, auc_ci = bootstrap_ci_metric(y_test, y_pred, y_probs, roc_auc_score)
    brier, brier_ci = bootstrap_brier(y_test, y_probs)

    results.append([name,
                    f"{acc:.3f} ({acc_ci[0]:.3f}–{acc_ci[1]:.3f})",
                    f"{prec:.3f} ({prec_ci[0]:.3f}–{prec_ci[1]:.3f})",
                    f"{rec:.3f} ({rec_ci[0]:.3f}–{rec_ci[1]:.3f})",
                    f"{f1:.3f} ({f1_ci[0]:.3f}–{f1_ci[1]:.3f})",
                    f"{auc:.3f} ({auc_ci[0]:.3f}–{auc_ci[1]:.3f})",
                    f"{brier:.3f} ({brier_ci[0]:.3f}–{brier_ci[1]:.3f})"])

# ==============================
# 3. Print summary table
# ==============================
import pandas as pd
df_results = pd.DataFrame(results,
                          columns=["Model", "Accuracy", "Precision", "Recall",
                                   "F1-Score", "ROC-AUC", "Brier Score"])
print("\n=== Model Performance with 95% Confidence Intervals ===")
print(df_results)

# ==============================
# 4. Calibration curves
# ==============================
plt.figure(figsize=(6,5))
for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curves")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# 5. Decision Curve Analysis
# ==============================
plt.figure(figsize=(6,5))
thresholds = np.linspace(0.01, 0.99, 50)

for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:, 1]
    mean_nb, lower, upper = bootstrap_dca(y_test, y_probs, thresholds=thresholds)
    plt.plot(thresholds, mean_nb, label=name)
    plt.fill_between(thresholds, lower, upper, alpha=0.2)

plt.plot(thresholds, [0]*len(thresholds), linestyle="--", color="black", label="Treat None")
plt.plot(thresholds, [np.mean(y_test)]*len(thresholds), linestyle="--", color="gray", label="Treat All")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (95% CI)")
plt.legend()
plt.grid(True)
plt.show()

```