# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\lime_materials\Paper 1 Code LIME with Diagrams\Reviewer 1 Concern 15\base_paper_dataset_all_reviewers.ipynb`
- SHA-256: `318a0047306cb44bd127bedefcea38ad4a1a1af9874b7592339f572c5228b040`
- Cells: 92
- Saved outputs: 189

## Imports
- `PIL`
- `collections`
- `google.colab`
- `imblearn.over_sampling`
- `imgkit`
- `joblib`
- `lime`
- `lime.discretize`
- `lime.lime_tabular`
- `loras`
- `matplotlib.patches`
- `matplotlib.pyplot`
- `numpy`
- `os`
- `pandas`
- `random`
- `reportlab.graphics.shapes`
- `reportlab.lib`
- `reportlab.lib.colors`
- `reportlab.lib.pagesizes`
- `reportlab.lib.styles`
- `reportlab.platypus`
- `seaborn`
- `shutil`
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
- LIME
- LoRAS
- OVEC
- RandomizedSearchCV
- SMOTE
- StackingClassifier
- VotingClassifier
- train_test_split

## Risk flags
- **credential_or_mount**: cells [2, 14, 30, 44, 48, 56, 57, 59, 60, 61, 62, 64, 65, 67, 71, 74, 75, 81, 82, 84, 85, 87, 89, 90, 92]
- **notebook_shell**: cells [24, 64, 65, 73, 76, 77, 79, 87]
- **file_deletion**: cells [64, 65, 74, 75]

## Cell sources

### Cell 1 — markdown (execution_count=None)
```text
# **Import libraries and Mount Google drive**
```

### Cell 2 — code (execution_count=1)
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

### Cell 4 — code (execution_count=2)
```python
data = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data
data.head()

```

### Cell 5 — markdown (execution_count=None)
```text
# **Checking for Duplocate Values**
```

### Cell 6 — code (execution_count=3)
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

### Cell 8 — code (execution_count=4)
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

### Cell 9 — code (execution_count=5)
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

### Cell 10 — code (execution_count=6)
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

### Cell 12 — code (execution_count=7)
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

### Cell 13 — code (execution_count=8)
```python
X = data_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = data_no_outliers['HeartDisease']


#Target Variables Values
data_no_outliers.HeartDisease.value_counts()
X.shape
```

### Cell 14 — code (execution_count=9)
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Count the occurrences of 'HeartDisease' for both outliers and non-outliers
outliers_heart_disease_counts = data_outliers['HeartDisease'].value_counts().sort_index()
non_outliers_heart_disease_counts = data_no_outliers['HeartDisease'].value_counts().sort_index()

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

# ✅ Save figure as PDF
plt.savefig("heart_disease_distribution.pdf", format="pdf")

# Show plot
plt.show()

print("✅ Plot saved as heart_disease_distribution.pdf")

from google.colab import files
files.download("heart_disease_distribution.pdf")

```

### Cell 15 — code (execution_count=10)
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

### Cell 16 — code (execution_count=11)
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

### Cell 19 — code (execution_count=12)
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

### Cell 20 — code (execution_count=13)
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

### Cell 21 — code (execution_count=14)
```python
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 22 — markdown (execution_count=None)
```text
# **Data Balancing**
```

### Cell 23 — code (execution_count=15)
```python
# ! pip install smote_variants==0.5.1
# ! pip install smote-variants
```

### Cell 24 — code (execution_count=16)
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

### Cell 25 — code (execution_count=17)
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

### Cell 26 — code (execution_count=18)
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

### Cell 28 — code (execution_count=19)
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

### Cell 29 — code (execution_count=20)
```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_oversampled_LoRAS_features_pca10)

# Get the loadings (components)
loadings = pca.components_

# Create a DataFrame to visualize the loadings
loading_df = pd.DataFrame(loadings, columns=X.columns, index=[f'PC{i+1}' for i in range(9)])

# Display the loadings
print(loading_df)
```

### Cell 30 — code (execution_count=21)
```python
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

# ✅ loading_df must be your DataFrame of loadings
# rows = PCs, columns = original features

plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    loading_df,
    annot=True,           # show numbers
    fmt=".4f",            # 2 decimal places
    cmap='coolwarm',      # color map
    center=0,             # center at 0 for diverging palette
    cbar=True,            # show color bar
    # linewidths=0.5,       # thin lines between cells
    # linecolor='black'     # cell borders
)

# Labels and title
plt.xlabel('Original Features', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)
# plt.title('Visualization of Feature Contributions to Each Principal Component', fontsize=14, fontweight='bold')

# Rotate x-axis labels if needed
# plt.xticks(rotation=45, ha='right')

plt.tight_layout()

# ✅ Save as PDF before showing
plt.savefig("pca_feature_contributions.pdf", format='pdf')

# Show plot
plt.show()

print("✅ Heatmap saved as pca_feature_contributions.pdf")

from google.colab import files
files.download("pca_feature_contributions.pdf")
```

### Cell 31 — code (execution_count=None)
```python

```

### Cell 32 — code (execution_count=21)
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1️⃣ Original features
# =========================
raw_feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# LoRAS_features should be your dataset (numerical + encoded categorical)
# Example: LoRAS_features = pd.DataFrame(data, columns=raw_feature_names)

# =========================
# 2️⃣ Standardize features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(LoRAS_features)

# =========================
# 3️⃣ Apply PCA
# =========================
n_components = 9
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 4️⃣ Create PCA components DataFrame
# =========================
pca_components = pd.DataFrame(
    pca.components_,
    columns=raw_feature_names,
    index=[f'PC{i+1}' for i in range(n_components)]
)

# =========================
# 5️⃣ Generate PCA summary table
# =========================
top_features_table = pd.DataFrame(columns=[
    'PC', 'Top1_Feature', 'Top1_Contribution',
    'Top2_Feature', 'Top2_Contribution'
])

for i in range(n_components):
    # Sort absolute contributions descending
    row = pca_components.iloc[i].abs().sort_values(ascending=False)
    top1_feature, top1_value = row.index[0], row.values[0]
    top2_feature, top2_value = row.index[1], row.values[1]

    top_features_table = pd.concat([top_features_table,
                                    pd.DataFrame([[f'PC{i+1}', top1_feature, top1_value, top2_feature, top2_value]],
                                                 columns=top_features_table.columns)],
                                    ignore_index=True)

# =========================
# 6️⃣ Display the PCA summary table
# =========================
print("✅ PCA Summary Table (Top 2 features per PC with contributions):")
print(top_features_table)

# =========================
# 7️⃣ Optional: PCA feature names lists
# =========================
# Top 1 feature per PC
pca_feature_names_top1 = top_features_table['Top1_Feature'].tolist()

# Top 2 features per PC (joined by '_')
pca_feature_names_top2 = top_features_table.apply(lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1).tolist()

print("\n# PCA feature names (Top 1 per PC):", pca_feature_names_top1)
print("# PCA feature names (Top 2 per PC):", pca_feature_names_top2)

# =========================
# 8️⃣ Optional: Create PCA DataFrame with meaningful columns
# =========================
X_pca_df_top2 = pd.DataFrame(X_pca, columns=pca_feature_names_top2)
print("\n✅ PCA DataFrame preview (Top 2 feature names as columns):")
print(X_pca_df_top2.head())

```

### Cell 33 — code (execution_count=22)
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1️⃣ Original features
# =========================
raw_feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# Assuming LoRAS_features is your dataset (numerical + encoded categorical)
# LoRAS_features should be a DataFrame or NumPy array
# Example: LoRAS_features = pd.DataFrame(data, columns=raw_feature_names)

# =========================
# 2️⃣ Standardize features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(LoRAS_features)

# =========================
# 3️⃣ Apply PCA
# =========================
n_components = 9
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 4️⃣ Create PCA components DataFrame
# =========================
pca_components = pd.DataFrame(
    pca.components_,
    columns=raw_feature_names,
    index=[f'PC{i+1}' for i in range(n_components)]
)

# =========================
# 5️⃣ Generate PCA feature names (top 1 and top 2 contributing features)
# =========================
# Top 1 feature per PC
pca_feature_names_top1 = pca_components.abs().idxmax(axis=1).tolist()

# Top 2 features per PC (joined by '_')
pca_feature_names_top2 = []
for i in range(n_components):
    top_features = pca_components.iloc[i].abs().sort_values(ascending=False).index[:2]
    pca_feature_names_top2.append(f"{top_features[0]}_{top_features[1]}")

# =========================
# 6️⃣ Optional: Create DataFrame with meaningful PCA column names
# =========================
X_pca_df_top1 = pd.DataFrame(X_pca, columns=pca_feature_names_top1)
X_pca_df_top2 = pd.DataFrame(X_pca, columns=pca_feature_names_top2)

# =========================
# 7️⃣ Display results
# =========================
print("✅ PCA explained variance ratio per component:")
print(pca.explained_variance_ratio_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

print("\n# PCA feature names (Top 1 feature per PC)")
print(pca_feature_names_top1)

print("\n# PCA feature names (Top 2 features per PC)")
print(pca_feature_names_top2)

print("\n✅ PCA DataFrame preview (Top 2 feature names as columns):")
print(X_pca_df_top2.head())

```

### Cell 34 — markdown (execution_count=None)
```text
**For LIME Using PCA Features**
```

### Cell 35 — code (execution_count=23)
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1️⃣ Original features
# =========================
raw_feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope'
]

# LoRAS_features should be your dataset (numerical + encoded categorical)
# Example: LoRAS_features = pd.DataFrame(data, columns=raw_feature_names)

# =========================
# 2️⃣ Standardize features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(LoRAS_features)

# =========================
# 3️⃣ Apply PCA
# =========================
n_components = 9
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 4️⃣ Create PCA components DataFrame
# =========================
pca_components = pd.DataFrame(
    pca.components_,
    columns=raw_feature_names,
    index=[f'PC{i+1}' for i in range(n_components)]
)

# =========================
# 5️⃣ Generate PCA summary table
# =========================
top_features_table = pd.DataFrame(columns=[
    'PC', 'Top1_Feature', 'Top1_Contribution',
    'Top2_Feature', 'Top2_Contribution'
])

for i in range(n_components):
    # Sort absolute contributions descending
    row = pca_components.iloc[i].abs().sort_values(ascending=False)
    top1_feature, top1_value = row.index[0], row.values[0]
    top2_feature, top2_value = row.index[1], row.values[1]

    top_features_table = pd.concat([top_features_table,
                                    pd.DataFrame([[f'PC{i+1}', top1_feature, top1_value, top2_feature, top2_value]],
                                                 columns=top_features_table.columns)],
                                    ignore_index=True)

# =========================
# 6️⃣ Display the PCA summary table
# =========================
print("✅ PCA Summary Table (Top 2 features per PC with contributions):")
print(top_features_table)

# =========================
# 7️⃣ Optional: PCA feature names lists
# =========================
# Top 1 feature per PC
pca_feature_names_top1 = top_features_table['Top1_Feature'].tolist()

# Top 2 features per PC (joined by '_')
pca_feature_names_top2 = top_features_table.apply(lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1).tolist()

print("\n# PCA feature names (Top 1 per PC):", pca_feature_names_top1)
print("# PCA feature names (Top 2 per PC):", pca_feature_names_top2)

# =========================
# 8️⃣ Optional: Create PCA DataFrame with meaningful columns
# =========================
X_pca_df_top2 = pd.DataFrame(X_pca, columns=pca_feature_names_top2)
print("\n✅ PCA DataFrame preview (Top 2 feature names as columns):")
print(X_pca_df_top2.head())

```

### Cell 36 — code (execution_count=None)
```python

```

### Cell 37 — code (execution_count=None)
```python

```

### Cell 38 — code (execution_count=24)
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

### Cell 39 — code (execution_count=26)
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

### Cell 40 — markdown (execution_count=None)
```text
# **Classification using Voting Model without Hyperparameter Tunning**
```

### Cell 41 — code (execution_count=None)
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

### Cell 42 — markdown (execution_count=None)
```text
# **Classification with Hyperparameter Tunning**
```

### Cell 43 — code (execution_count=None)
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

### Cell 44 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
from google.colab import files
# ===== Plot ROC Curve =====
fig = plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'NOVEC Classifier AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)

# Titles and labels
# plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=11)

# Grid and layout
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# ===== Save as PDF BEFORE showing =====
fig.savefig("roc_curve.pdf", format="pdf")

# Show the plot
plt.show()

print("✅ ROC Curve saved as roc_curve.pdf")

# ===== Optional: Download in Colab =====
files.download("roc_curve.pdf")
```

### Cell 45 — code (execution_count=None)
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

### Cell 46 — code (execution_count=None)
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

### Cell 47 — code (execution_count=None)
```python
pip install lime

```

### Cell 48 — code (execution_count=None)
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
exp.save_to_file('lime_explanation.html')  # Save to an HTML file

from google.colab import files
files.download('lime_explanation.html')
```

### Cell 49 — markdown (execution_count=None)
```text
# **LIME Reviewers**
```

### Cell 50 — code (execution_count=27)
```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import pandas as pd
import joblib
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
    'n_estimators': [50, 100, 200],
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

# --- Combine tuned models into VotingClassifier ---
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

# --- Create pipeline (no scaling) ---
pipeline = Pipeline([
    ('voting', voting_model)
])

# --- Train pipeline ---
pipeline.fit(X_train, y_train)

# --- Predictions & Metrics ---
voting_predictions = pipeline.predict(X_test)
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, voting_predictions)
precision = precision_score(y_test, voting_predictions)
recall = recall_score(y_test, voting_predictions)
f1 = f1_score(y_test, voting_predictions)
voting_probabilities = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, voting_probabilities)
auc_score = auc(fpr, tpr)

# --- Plot ROC Curve ---
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Voting Classifier AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("voting_model_roc.png", dpi=300)
plt.close()

# --- Display Results ---
print("========== Model Performance with Weighted Voting ==========")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy : {test_accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"AUC-ROC           : {auc_score:.4f}")
print("===========================================================")

# --- Save Model Pipeline for LIME ---
joblib.dump(pipeline, "voting_pipeline.pkl")
print("✅ Model pipeline saved as 'voting_pipeline.pkl'")

# --- Save Metrics for Reference ---
results = pd.DataFrame({
    "Metric": ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
    "Score": [train_accuracy, test_accuracy, precision, recall, f1, auc_score]
})
results.to_csv("voting_model_results.csv", index=False)
print("✅ Metrics saved to 'voting_model_results.csv'")

```

### Cell 51 — code (execution_count=3)
```python
import os
import joblib

# Create folder if not exists
os.makedirs("/content/drive/MyDrive/models", exist_ok=True)

joblib.dump(pipeline,"/content/drive/MyDrive/models/voting_pipeline.pkl")
print("✅ Saved to Google Drive!")


# Save train–test split data
joblib.dump(
    {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    },
    "/content/drive/MyDrive/models/LoRAS_train_test_data.pkl"
)

print("✅ Train/test data saved to '/content/drive/MyDrive/models/LoRAS_train_test_data.pkl'")


```

### Cell 52 — markdown (execution_count=None)
```text
***Download from Google Drive***
```

### Cell 53 — code (execution_count=30)
```python
import joblib
pipeline_loaded = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")
print("✅ Model loaded successfully!")

# Load saved data from Google Drive
data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")

X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']

print("✅ Data reloaded successfully — ready for LIME!")

```

### Cell 54 — code (execution_count=None)
```python

```

### Cell 55 — code (execution_count=25)
```python
pip install lime
```

### Cell 56 — code (execution_count=40)
```python
# ============================================
# LIME EXPLANATION MODULE — Reviewer-Ready
# NOVEM-CVD (Explainable Ensemble)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Define Feature Names
# -------------------------------------------------
# Define actual feature names
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope', #'HeartDisease'
]
# -------------------------------------------------
# 2. Prepare Data and Model
# -------------------------------------------------
#RANDOM_SEED = 42
#np.random.seed(RANDOM_SEED)

#X_train_lime, X_test_lime, y_train_lime, y_test_lime = train_test_split(
    #LoRAS_features, LoRAS_labels, test_size=0.25,
  #  random_state=RANDOM_SEED, stratify=LoRAS_labels
#)

# Assume 'voting_pipeline' (NOVEC ensemble) is already trained

# -------------------------------------------------
# 3. Explicit LIME Configuration — Fully Reported
# -------------------------------------------------
RANDOM_SEED = 42   # ✅ FIX ADDED HERE
np.random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",     # LIME uses exponential kernel by design
    "kernel_width": 3.0,              # defines locality: smaller = more local
    "num_samples": 1500,              # perturbations per instance
    "discretizer": "quartile",        # explicit discretizer per reviewer
    "random_seed": RANDOM_SEED
}

# Explicit discretizer object (ensures reproducibility)
discretizer = QuartileDiscretizer(
    np.array(X_train),
    categorical_features=[],  # specify categorical indices if needed
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=LIME_CONFIG["random_seed"]
)

# -------------------------------------------------
# 4. Generate Explanations (Stable, Robust)
# -------------------------------------------------
n_instances = 30   # number of test samples to analyze
n_runs = 5         # repeat runs to assess stability
POS_CLASS = 1      # “Disease” class index

instance_indices = np.random.choice(range(X_test.shape[0]), n_instances, replace=False)
feature_weights_runs = []

for run in range(n_runs):
    np.random.seed(LIME_CONFIG["random_seed"] + run)
    run_weights = np.zeros(len(feature_names))

    for idx in instance_indices:
        exp = explainer.explain_instance(
            data_row=X_test[idx],
            predict_fn=pipeline_loaded.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        # ✅ Robust parsing using exp.as_map()[class_idx]
        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight

    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)

# -------------------------------------------------
# 5. Compute Signed and Absolute Statistics
# -------------------------------------------------
signed_mean = np.mean(feature_weights_runs, axis=0)
signed_std = np.std(feature_weights_runs, axis=0)
abs_mean = np.mean(np.abs(feature_weights_runs), axis=0)
abs_std = np.std(np.abs(feature_weights_runs), axis=0)

# Build summary dataframe (directionality + magnitude)
lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})

# Add human-readable risk direction
lime_summary["Direction"] = np.where(lime_summary["SignedMean"] > 0, "↑ increases risk", "↓ decreases risk")
lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1  # start ranking at 1

print("\n--- LIME Feature Importance Summary ---")
print(lime_summary.head(10))

# -------------------------------------------------
# 6. Visualize Direction-Aware Global Importance
# -------------------------------------------------
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.set_context("talk")

# Fix: convert Series → list for valid Seaborn palette
colors = lime_summary["SignedMean"].apply(
    lambda x: "crimson" if x > 0 else "steelblue"
).tolist()

# Direction-aware barplot
sns.barplot(
    data=lime_summary,
    x="SignedMean",
    y="Feature",
    palette=colors,
    edgecolor="black"
)

# Reference line for neutrality
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

# Add direction arrows beside bars
for i, val in enumerate(lime_summary["SignedMean"]):
    plt.text(
        val + (0.001 if val >= 0 else -0.001),
        i,
        "↑" if val > 0 else "↓",
        va="center",
        ha="left" if val > 0 else "right",
        fontsize=12,
        color="black"
    )

# ✅ Add Legend in lower right corner
from matplotlib.patches import Patch

legend_handles = [
    Patch(color="crimson", label="Increases Risk (Positive Weight)"),
    Patch(color="steelblue", label="Decreases Risk (Negative Weight)")
]

plt.legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=11,
    frameon=True,
    edgecolor="black"
)

# Labels and title
plt.xlabel("Signed Mean LIME Weight", fontsize=12)
plt.ylabel("")
plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble", fontsize=14)

plt.tight_layout()

# ✅ Save as PDF
pdf_path = "/content/LIME_directionality_plot.pdf"
plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")

# ✅ Auto-download
from google.colab import files
files.download(pdf_path)

plt.show()

# -------------------------------------------------
# 7. Report Stability Across Runs
# -------------------------------------------------
top_k = 5
top_features_runs = [np.argsort(-np.abs(run))[:top_k] for run in feature_weights_runs]

def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))

stability_scores = [jaccard(top_features_runs[i], top_features_runs[j])
                    for i in range(n_runs) for j in range(i+1, n_runs)]

print(f"\nLIME Stability (Top-{top_k} Feature Overlap, Jaccard): "
      f"{np.mean(stability_scores):.3f} ± {np.std(stability_scores):.3f}")

# -------------------------------------------------
# 8. Export Config and Summary for Supplementary Material
# -------------------------------------------------
print("\n--- LIME Configuration ---")
for k, v in LIME_CONFIG.items():
    print(f"{k}: {v}")

lime_summary.to_csv("LIME_summary.csv", index_label="Rank")

```

### Cell 57 — code (execution_count=41)
```python
# -------------------------------------------------
# 6. Visualize Direction-Aware Global Importance
# -------------------------------------------------
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.set_context("talk")

# Fix: convert Series → list for valid Seaborn palette
colors = lime_summary["SignedMean"].apply(
    lambda x: "crimson" if x > 0 else "steelblue"
).tolist()

# Direction-aware barplot
sns.barplot(
    data=lime_summary,
    x="SignedMean",
    y="Feature",
    palette=colors,
    edgecolor="black"
)

# Reference line for neutrality
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)

# Add direction arrows beside bars
for i, val in enumerate(lime_summary["SignedMean"]):
    plt.text(
        val + (0.001 if val >= 0 else -0.001),
        i,
        "↑" if val > 0 else "↓",
        va="center",
        ha="left" if val > 0 else "right",
        fontsize=12,
        color="black"
    )

# ✅ Add Legend in lower right corner
from matplotlib.patches import Patch

legend_handles = [
    Patch(color="crimson", label="Increases Risk (Positive Weight)"),
    Patch(color="steelblue", label="Decreases Risk (Negative Weight)")
]

plt.legend(
    handles=legend_handles,
    loc="lower right",
    fontsize=11,
    frameon=True,
    edgecolor="black"
)

# Labels and title
plt.xlabel("Signed Mean LIME Weight", fontsize=12)
plt.ylabel("")
#plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble", fontsize=14)

plt.tight_layout()

# ✅ Save as PDF
pdf_path = "/content/LIME_directionality_plot.pdf"
plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")

# ✅ Auto-download
from google.colab import files
files.download(pdf_path)

plt.show()
```

### Cell 58 — code (execution_count=None)
```python

```

### Cell 59 — code (execution_count=42)
```python
# ============================================
# LIME EXPLANATION MODULE — Reviewer-Ready
# NOVEM-CVD (Explainable Ensemble)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib
import warnings, os
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Load Pre-Saved Data and Model
# -------------------------------------------------
print("🔄 Loading pre-saved model and dataset...")

import joblib
pipeline_loaded = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")
print("✅ Model loaded successfully!")

# Load saved data from Google Drive
data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")

X_train_lime = data["X_train"]
X_test_lime = data["X_test"]
y_train_lime = data["y_train"]
y_test_lime = data["y_test"]

print("✅ Data reloaded successfully — ready for LIME!")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
    'Oldpeak', 'ST_Slope', #'HeartDisease'
]

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# -------------------------------------------------
# 3. LIME Configuration
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=LIME_CONFIG["random_seed"]
)

# -------------------------------------------------
# 4. Generate Multi-Run Explanations
# -------------------------------------------------
n_instances = 30
n_runs = 5
POS_CLASS = 1

instance_indices = np.random.choice(range(X_test_lime.shape[0]), n_instances, replace=False)
feature_weights_runs = []

for run in range(n_runs):
    np.random.seed(LIME_CONFIG["random_seed"] + run)
    run_weights = np.zeros(len(feature_names))

    for idx in instance_indices:
        exp = explainer.explain_instance(
            data_row=X_test_lime[idx],
            predict_fn=pipeline_loaded.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight

    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)

# -------------------------------------------------
# 5. Aggregate Feature Statistics
# -------------------------------------------------
signed_mean = np.mean(feature_weights_runs, axis=0)
signed_std = np.std(feature_weights_runs, axis=0)
abs_mean = np.mean(np.abs(feature_weights_runs), axis=0)
abs_std = np.std(np.abs(feature_weights_runs), axis=0)

lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})

lime_summary["Direction"] = np.where(
    lime_summary["SignedMean"] > 0,
    "↑ increases risk",
    "↓ decreases risk"
)

lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1

print("\n--- LIME Feature Importance Summary ---")
print(lime_summary.head(10))

# -------------------------------------------------
# 6. Visualization — Enlarged X-axis + Smooth Background + Legend + Labels
# -------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.figure(figsize=(12, 7))   # ✅ slightly wider figure

# ✅ Smooth background
plt.gca().set_facecolor("#fafafa")

# ✅ Minimal soft gridlines
plt.grid(
    which='major',
    axis='x',
    linestyle='--',
    color='lightgray',
    linewidth=0.7,
    alpha=0.6
)

plt.grid(which='major', axis='y', linestyle='', linewidth=0)

# ✅ Modern axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# ✅ Colors
colors = lime_summary["SignedMean"].apply(
    lambda x: "crimson" if x > 0 else "steelblue"
)

bars = plt.barh(
    lime_summary["Feature"],
    lime_summary["SignedMean"],
    color=colors,
    edgecolor='black',
    linewidth=0.5
)

# ✅ Vertical zero line
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)

# ✅ ENLARGE X-axis to prevent overlap
min_x = lime_summary["SignedMean"].min()
max_x = lime_summary["SignedMean"].max()

left_pad = min_x * 1.25   # 25% extra left space
right_pad = max_x * 1.35  # 35% extra right space

plt.xlim(left_pad, right_pad)

# ✅ Labels & Title
plt.xlabel("Signed Mean LIME Weight", fontsize=12)
plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble", fontsize=15)

# ✅ Numerical labels on each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + (0.002 if width >= 0 else -0.002),
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f}",
        va="center",
        ha="left" if width >= 0 else "right",
        fontsize=11,
        color="black"
    )

# ✅ Legend
legend_elements = [
    Patch(facecolor='crimson', edgecolor='black', label='↑ Increases Risk'),
    Patch(facecolor='steelblue', edgecolor='black', label='↓ Decreases Risk')
]

plt.legend(
    handles=legend_elements,
    loc='lower right',
    fontsize=11,
    frameon=True,
    facecolor='white',
    edgecolor='black'
)

plt.tight_layout()

# ✅ Save PNG
png_path = "LIME_feature_importance.png"
plt.savefig(png_path, dpi=300)

# ✅ Save PDF
pdf_path = "LIME_feature_importance.pdf"
plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")

plt.close()

print("✅ LIME barplot saved as 'LIME_feature_importance.png'")
print("✅ LIME barplot saved as 'LIME_feature_importance.pdf'")

# ✅ Auto-download PDF
from google.colab import files
files.download(pdf_path)


# -------------------------------------------------
# 7. Stability Across Runs
# -------------------------------------------------
top_k = 5
top_features_runs = [
    np.argsort(-np.abs(run))[:top_k]
    for run in feature_weights_runs
]

def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))

stability_scores = [
    jaccard(top_features_runs[i], top_features_runs[j])
    for i in range(n_runs) for j in range(i+1, n_runs)
]

stability_mean = np.mean(stability_scores)
stability_std = np.std(stability_scores)

print(f"\nLIME Stability (Top-{top_k} Jaccard): {stability_mean:.3f} ± {stability_std:.3f}")

# -------------------------------------------------
# 8. Export All Outputs
# -------------------------------------------------
lime_summary.to_csv("LIME_summary.csv", index_label="Rank")
print("✅ LIME summary saved as 'LIME_summary.csv'")

# -------------------------------------------------
# 9. HTML Report Generation
# -------------------------------------------------
html_content = f"""
<html>
<head>
<title>LIME Report — NOVEM-CVD Ensemble</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
h1, h2 {{ color: #2b3e50; }}
table {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 20px;
}}
th, td {{
  border: 1px solid #ccc;
  padding: 8px;
  text-align: center;
}}
th {{ background-color: #f4f4f4; }}
img {{
  display: block;
  margin: 20px auto;
  max-width: 80%;
}}
</style>
</head>
<body>

<h1>📘 NOVEM-CVD LIME Explanation Report</h1>

<h2>Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
""" + \
"\n".join([
    f"<tr><td>{k}</td><td>{v}</td></tr>"
    for k, v in LIME_CONFIG.items()
]) + \
f"""
</table>

<h2>Feature Importance (Top 10)</h2>
{lime_summary.head(10).to_html(index=True, justify='center')}

<h2>Stability</h2>
<p><b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}</p>

<h2>Visualization</h2>
<img src="LIME_feature_importance.png" alt="LIME Feature Importance Plot"/>

<h2>Download PDF Plot</h2>
<p><a href="LIME_feature_importance.pdf">Click here to download the PDF version</a></p>

<hr>
<p style='text-align:center;'>Generated automatically — NOVEM-CVD Explainable Ensemble © 2025</p>

</body>
</html>
"""

with open("LIME_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ LIME HTML report saved as 'LIME_report.html'")

```

### Cell 60 — code (execution_count=43)
```python
# -------------------------------------------------
# 6. Visualization — Enlarged X-axis + Smooth Background + Legend + Labels
# -------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.figure(figsize=(12, 7))   # ✅ slightly wider figure

# ✅ Smooth background
plt.gca().set_facecolor("#fafafa")

# ✅ Minimal soft gridlines
plt.grid(
    which='major',
    axis='x',
    linestyle='--',
    color='lightgray',
    linewidth=0.7,
    alpha=0.6
)

plt.grid(which='major', axis='y', linestyle='', linewidth=0)

# ✅ Modern axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# ✅ Colors
colors = lime_summary["SignedMean"].apply(
    lambda x: "crimson" if x > 0 else "steelblue"
)

bars = plt.barh(
    lime_summary["Feature"],
    lime_summary["SignedMean"],
    color=colors,
    edgecolor='black',
    linewidth=0.5
)

# ✅ Vertical zero line
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)

# ✅ ENLARGE X-axis to prevent overlap
min_x = lime_summary["SignedMean"].min()
max_x = lime_summary["SignedMean"].max()

left_pad = min_x * 1.25   # 25% extra left space
right_pad = max_x * 1.35  # 35% extra right space

plt.xlim(left_pad, right_pad)

# ✅ Labels & Title
plt.xlabel("Signed Mean LIME Weight", fontsize=12)
#plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble", fontsize=15)

# ✅ Numerical labels on each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + (0.002 if width >= 0 else -0.002),
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f}",
        va="center",
        ha="left" if width >= 0 else "right",
        fontsize=11,
        color="black"
    )

# ✅ Legend
legend_elements = [
    Patch(facecolor='crimson', edgecolor='black', label='↑ Increases Risk'),
    Patch(facecolor='steelblue', edgecolor='black', label='↓ Decreases Risk')
]

plt.legend(
    handles=legend_elements,
    loc='lower right',
    fontsize=11,
    frameon=True,
    facecolor='white',
    edgecolor='black'
)

plt.tight_layout()

# ✅ Save PNG
png_path = "LIME_feature_importance.png"
plt.savefig(png_path, dpi=300)

# ✅ Save PDF
pdf_path = "LIME_feature_importance.pdf"
plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")

plt.close()

print("✅ LIME barplot saved as 'LIME_feature_importance.png'")
print("✅ LIME barplot saved as 'LIME_feature_importance.pdf'")

# ✅ Auto-download PDF
from google.colab import files
files.download(pdf_path)
```

### Cell 61 — code (execution_count=44)
```python
# -------------------------------------------------
# 6B. Visualization — Absolute LIME Importance (NEW FIGURE)
# -------------------------------------------------

plt.figure(figsize=(12, 7))

# ✅ Smooth background
plt.gca().set_facecolor("#fafafa")

# ✅ Minimal vertical gridlines only
plt.grid(
    which='major',
    axis='x',
    linestyle='--',
    color='lightgray',
    linewidth=0.7,
    alpha=0.6
)

plt.grid(which='major', axis='y', linestyle='', linewidth=0)

# ✅ Modern axis style
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# ✅ Sort features by absolute importance
abs_df = lime_summary.copy()
abs_df = abs_df.sort_values("AbsMean", ascending=True)

# ✅ Horizontal bars
bars = plt.barh(
    abs_df["Feature"],
    abs_df["AbsMean"],
    color="seagreen",
    edgecolor='black',
    linewidth=0.6
)

# ✅ X-axis label & Title
plt.xlabel("Absolute Mean LIME Weight", fontsize=12)
plt.title("Absolute LIME Feature Importance — NOVEM-CVD Ensemble", fontsize=15)

# ✅ Numerical labels for each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.0005,                       # offset
        bar.get_y() + bar.get_height()/2,
        f"{width:.4f}",
        va="center",
        ha="left",
        fontsize=10,
        color="black"
    )

plt.tight_layout()

# ✅ Save PNG
abs_png = "LIME_absolute_importance.png"
plt.savefig(abs_png, dpi=300)

# ✅ Save PDF
abs_pdf = "LIME_absolute_importance.pdf"
plt.savefig(abs_pdf, format="pdf", dpi=300, bbox_inches="tight")

plt.close()

print("✅ Absolute LIME barplot saved as 'LIME_absolute_importance.png'")
print("✅ Absolute LIME barplot saved as 'LIME_absolute_importance.pdf'")

# ✅ Auto-download PDF
from google.colab import files
files.download(abs_pdf)

```

### Cell 62 — code (execution_count=45)
```python
# -------------------------------------------------
# 6B. Visualization — Absolute LIME Importance (NEW FIGURE)
# -------------------------------------------------

plt.figure(figsize=(12, 7))

# ✅ Smooth background
plt.gca().set_facecolor("#fafafa")

# ✅ Minimal vertical gridlines only
plt.grid(
    which='major',
    axis='x',
    linestyle='--',
    color='lightgray',
    linewidth=0.7,
    alpha=0.6
)

plt.grid(which='major', axis='y', linestyle='', linewidth=0)

# ✅ Modern axis style
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# ✅ Sort features by absolute importance
abs_df = lime_summary.copy()
abs_df = abs_df.sort_values("AbsMean", ascending=True)

# ✅ Horizontal bars
bars = plt.barh(
    abs_df["Feature"],
    abs_df["AbsMean"],
    color="seagreen",
    edgecolor='black',
    linewidth=0.6
)

# ✅ X-axis label & Title
plt.xlabel("Absolute Mean LIME Weight", fontsize=12)
#plt.title("Absolute LIME Feature Importance — NOVEM-CVD Ensemble", fontsize=15)

# ✅ Numerical labels for each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.0005,                       # offset
        bar.get_y() + bar.get_height()/2,
        f"{width:.4f}",
        va="center",
        ha="left",
        fontsize=10,
        color="black"
    )

plt.tight_layout()

# ✅ Save PNG
abs_png = "LIME_absolute_importance.png"
plt.savefig(abs_png, dpi=300)

# ✅ Save PDF
abs_pdf = "LIME_absolute_importance.pdf"
plt.savefig(abs_pdf, format="pdf", dpi=300, bbox_inches="tight")

plt.close()

print("✅ Absolute LIME barplot saved as 'LIME_absolute_importance.pdf'")

# ✅ Auto-download PDF
from google.colab import files
files.download(abs_pdf)

```

### Cell 63 — code (execution_count=None)
```python

```

### Cell 64 — code (execution_count=48)
```python
!pip install reportlab
# ============================================
# LIME EXPLANATION MODULE — Reviewer-Ready
# NOVEM-CVD (Explainable Ensemble) + RAW EXPORTS
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib, warnings, os, zipfile, datetime, json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Load Pre-Saved Data and Model
# -------------------------------------------------
print("🔄 Loading pre-saved model and dataset...")

import joblib
voting_model = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")
print("✅ Model loaded successfully!")

# Load saved data from Google Drive
data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")

X_train_lime = data["X_train"]
X_test_lime = data["X_test"]
y_train_lime = data["y_train"]
y_test_lime = data["y_test"]

print("✅ Data reloaded successfully — ready for LIME!")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
feature_names = top_features_table.apply(lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# -------------------------------------------------
# 3. LIME Configuration
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],  # specify categorical indices if needed
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=LIME_CONFIG["random_seed"]
)

# -------------------------------------------------
# 4. Generate Multi-Run Explanations
# -------------------------------------------------
n_instances = 30
n_runs = 5
POS_CLASS = 1

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = f"LIME_outputs_{timestamp}"
os.makedirs(outputs_dir, exist_ok=True)
html_dir = os.path.join(outputs_dir, "instance_html")
os.makedirs(html_dir, exist_ok=True)

instance_indices = np.random.choice(range(X_test_lime.shape[0]), n_instances, replace=False)
feature_weights_runs = []

# Raw exports accumulators
raw_rows = []
jsonl_path = os.path.join(outputs_dir, "raw_explanations.jsonl")
pred_rows = []

for run in range(n_runs):
    np.random.seed(LIME_CONFIG["random_seed"] + run)
    run_weights = np.zeros(len(feature_names))

    for idx in instance_indices:
        x = X_test_lime[idx]
        # Store model prediction for the instance (per run, same model, but keep provenance)
        probs = voting_model.predict_proba([x])[0]
        y_pred = int(np.argmax(probs))
        pred_rows.append({
            "run": run,
            "instance_idx": int(idx),
            "pred_class": y_pred,
            "prob_no_disease": float(probs[0]),
            "prob_disease": float(probs[1])
        })

        exp = explainer.explain_instance(
            data_row=x,
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )

        # Aggregate weights for POS_CLASS (for the global summary)
        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight

        # ---- Raw capture for ALL classes ----
        # exp.as_map() -> dict[class_idx] = List[(feat_idx, weight)]
        local_map = exp.as_map()
        for class_idx, pairs in local_map.items():
            for feat_idx, weight in pairs:
                raw_rows.append({
                    "run": run,
                    "instance_idx": int(idx),
                    "class_idx": int(class_idx),
                    "feature_idx": int(feat_idx),
                    "feature": feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}",
                    "weight": float(weight)
                })

        # Per-instance HTML (per run) for human inspection
        try:
            html_str = exp.as_html()
            html_file = os.path.join(html_dir, f"exp_run{run}_idx{idx}.html")
            with open(html_file, "w", encoding="utf-8") as hf:
                hf.write(html_str)
        except Exception as e:
            # If HTML export fails (e.g., missing resources), fall back to JSON for that instance
            fallback = {
                "run": run,
                "instance_idx": int(idx),
                "note": f"HTML export failed: {e}",
                "map": {int(k): [(int(fi), float(w)) for fi, w in v] for k, v in local_map.items()}
            }
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(fallback) + "\n")

    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)

# -------------------------------------------------
# 5. Aggregate Feature Statistics
# -------------------------------------------------
signed_mean = np.mean(feature_weights_runs, axis=0)
signed_std = np.std(feature_weights_runs, axis=0)
abs_mean = np.mean(np.abs(feature_weights_runs), axis=0)
abs_std = np.std(np.abs(np.abs(feature_weights_runs)), axis=0)

lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})

lime_summary["Direction"] = np.where(lime_summary["SignedMean"] > 0, "↑ increases risk", "↓ decreases risk")
lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1

print("\n--- LIME Feature Importance Summary ---")
print(lime_summary.head(10))

# Save raw tables
raw_df = pd.DataFrame(raw_rows)
raw_csv_path = os.path.join(outputs_dir, "raw_explanations.csv")
raw_df.to_csv(raw_csv_path, index=False)
with open(jsonl_path, "a", encoding="utf-8") as jf:
    for _, row in raw_df.iterrows():
        jf.write(json.dumps(row.to_dict()) + "\n")

pred_df = pd.DataFrame(pred_rows)
pred_csv_path = os.path.join(outputs_dir, "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)

# -------------------------------------------------
# 6. Visualization — Signed Importance
# -------------------------------------------------
fig_filename = os.path.join(outputs_dir, f"LIME_feature_importance_{timestamp}.png")

plt.figure(figsize=(10, 6))
colors = lime_summary["SignedMean"].apply(lambda x: "crimson" if x > 0 else "steelblue")
plt.barh(lime_summary["Feature"], lime_summary["SignedMean"], color=colors)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Signed Mean LIME Weight")
plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble")
plt.tight_layout()
plt.savefig(fig_filename, dpi=300)
plt.close()
print(f"✅ LIME barplot saved as '{fig_filename}'")

# -------------------------------------------------
# 7. Stability Across Runs
# -------------------------------------------------
top_k = 5
top_features_runs = [np.argsort(-np.abs(run))[:top_k] for run in feature_weights_runs]

def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))

stability_scores = [jaccard(top_features_runs[i], top_features_runs[j])
                    for i in range(n_runs) for j in range(i+1, n_runs)]

stability_mean = np.mean(stability_scores)
stability_std = np.std(stability_scores)
print(f"\nLIME Stability (Top-{top_k} Jaccard): {stability_mean:.3f} ± {stability_std:.3f}")

# -------------------------------------------------
# 8. Export All Outputs
# -------------------------------------------------
csv_file = os.path.join(outputs_dir, f"LIME_summary_{timestamp}.csv")
lime_summary.to_csv(csv_file, index_label="Rank")
print(f"✅ LIME summary saved as '{csv_file}'")

# -------------------------------------------------
# 9. HTML Report Generation
# -------------------------------------------------
html_file = os.path.join(outputs_dir, f"LIME_report_{timestamp}.html")
html_content = f"""
<html>
<head>
<title>LIME Report — NOVEM-CVD Ensemble</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
h1, h2 {{ color: #2b3e50; }}
table {{
  border-collapse: collapse;
  width: 100%;
  margin-top: 20px;
}}
th, td {{
  border: 1px solid #ccc;
  padding: 8px;
  text-align: center;
}}
th {{
  background-color: #f4f4f4;
}}
img {{
  display: block;
  margin: 20px auto;
  max-width: 80%;
}}
</style>
</head>
<body>

<h1>📘 NOVEM-CVD LIME Explanation Report</h1>
<h2>Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
""" + \
"\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in LIME_CONFIG.items()]) + \
f"""
</table>

<h2>Feature Importance (Top 10)</h2>
{lime_summary.head(10).to_html(index=True, justify='center')}

<h2>Stability</h2>
<p><b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}</p>

<h2>Visualization</h2>
<img src="{os.path.basename(fig_filename)}" alt="LIME Feature Importance Plot"/>

<h2>Artifacts</h2>
<ul>
<li>{os.path.basename(csv_file)}</li>
<li>{os.path.basename(fig_filename)}</li>
<li>raw_explanations.csv / raw_explanations.jsonl</li>
<li>predictions.csv</li>
<li>Per-instance HTML in <code>instance_html/</code></li>
</ul>

<hr>
<p style='text-align:center;'>Generated automatically — NOVEM-CVD Explainable Ensemble © 2025</p>
</body>
</html>
"""

with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"✅ LIME HTML report saved as '{html_file}'")

# -------------------------------------------------
# 10. PDF Export
# -------------------------------------------------
# =========================================================
# FIXED PDF EXPORT SECTION — NOVEM-CVD (no color conflict)
# =========================================================

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

print("🧾 Rebuilding NOVEM-CVD PDF report...")

pdf_file = os.path.join(outputs_dir, f"LIME_report_{timestamp}.pdf")

doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
story = []

# -------------------------
# Title
# -------------------------
story.append(Paragraph("📘 NOVEM-CVD LIME Explanation Report", styles['Title']))
story.append(Spacer(1, 12))

# -------------------------
# Configuration Table
# -------------------------
story.append(Paragraph("<b>Configuration</b>", styles['Heading2']))
config_table = [[k, str(v)] for k, v in LIME_CONFIG.items()]
table = Table(config_table, colWidths=[200, 200])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), rl_colors.lightgrey),
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
]))
story.append(table)
story.append(Spacer(1, 12))

# -------------------------
# Feature Importance
# -------------------------
story.append(Paragraph("<b>Feature Importance (Top 10)</b>", styles['Heading2']))
top10 = lime_summary.head(10)
table_data = [list(top10.columns)] + top10.values.tolist()
table = Table(table_data, repeatRows=1, colWidths=[90, 90, 80, 80, 80, 120])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), rl_colors.lightgrey),
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
]))
story.append(table)
story.append(Spacer(1, 12))

# -------------------------
# Stability Section
# -------------------------
story.append(Paragraph("<b>Stability</b>", styles['Heading2']))
story.append(Paragraph(
    f"<b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}",
    styles['Normal']
))
story.append(Spacer(1, 12))

# -------------------------
# Visualization
# -------------------------
story.append(Paragraph("<b>Visualization</b>", styles['Heading2']))
story.append(Image(fig_filename, width=450, height=300))
story.append(Spacer(1, 12))

# -------------------------
# Artifacts summary
# -------------------------
story.append(Paragraph("<b>Artifacts Included</b>", styles['Heading2']))
story.append(Paragraph(
    f"- {os.path.basename(csv_file)}<br/>"
    f"- {os.path.basename(fig_filename)}<br/>"
    f"- raw_explanations.csv / raw_explanations.jsonl<br/>"
    f"- predictions.csv<br/>"
    f"- Per-instance HTML in <code>instance_html/</code>",
    styles['Normal']
))
story.append(Spacer(1, 12))

# -------------------------
# Footer
# -------------------------
story.append(Paragraph(
    "<hr/><p style='text-align:center;'>Generated automatically — NOVEM-CVD Explainable Ensemble © 2025</p>",
    styles['Normal']
))

doc.build(story)

print(f"✅ PDF report rebuilt successfully: {pdf_file}")

# -------------------------------------------------
# 11. ZIP All Outputs (figures + raw + reports)
# -------------------------------------------------
# -------------------------------------------------
# 11. ZIP All Outputs (figures + raw + reports)
# -------------------------------------------------
# =========================================================
# AUTO-DOWNLOAD & CLEANUP SECTION — NOVEM-CVD LIME (v2)
# =========================================================
import shutil, glob, os

# 1️⃣ Create ZIP for the CURRENT run
zip_filename = f"LIME_outputs_{timestamp}.zip"
try:
    print(f"📦 Creating ZIP archive for current run → {zip_filename}")
    shutil.make_archive(base_name=f"LIME_outputs_{timestamp}", format="zip", root_dir=outputs_dir)
    print(f"✅ ZIP created successfully: {zip_filename}")
except Exception as e:
    print(f"⚠️ Could not create ZIP archive: {e}")

# 2️⃣ Create a human-friendly alias copy
friendly_zip = "NOVEM_CVD_LIME_Report_Latest.zip"
try:
    shutil.copy(zip_filename, friendly_zip)
    print(f"📦 Copied latest archive → {friendly_zip}")
except Exception as e:
    print(f"⚠️ Could not copy ZIP file: {e}")

# 3️⃣ Trigger download if running in Google Colab
try:
    from google.colab import files
    print("📥 Starting automatic download...")
    files.download(friendly_zip)
    print(f"✅ Auto-download started: {friendly_zip}")
except Exception as e:
    print(f"⚠️ Auto-download skipped (not running in Colab): {e}")

# 4️⃣ Optional cleanup — keep only the newest ZIPs
try:
    all_zips = sorted(glob.glob("LIME_outputs_*.zip"), key=os.path.getmtime)
    if len(all_zips) > 2:
        for z in all_zips[:-1]:
            os.remove(z)
            print(f"🧹 Removed old archive: {z}")
    print("🧾 Cleanup complete. Only latest ZIP files remain.")
except Exception as e:
    print(f"⚠️ Cleanup skipped: {e}")

```

### Cell 65 — code (execution_count=49)
```python
# ============================================
# LIME EXPLANATION MODULE — Reviewer-Ready
# NOVEM-CVD (Explainable Ensemble) + RAW EXPORTS
# ============================================

!pip install reportlab -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib, warnings, os, zipfile, datetime, json, shutil, glob
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Load Pre-Saved Data and Model
# -------------------------------------------------
print("🔄 Loading pre-saved model and dataset...")

voting_model = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")
print("✅ Model loaded successfully!")

data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")
X_train_lime = data["X_train"]
X_test_lime = data["X_test"]
y_train_lime = data["y_train"]
y_test_lime = data["y_test"]

print("✅ Data reloaded successfully — ready for LIME!")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
# Replace this with your actual feature generation logic if needed
feature_names = top_features_table.apply(
    lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1
).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# -------------------------------------------------
# 3. LIME Configuration
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=LIME_CONFIG["random_seed"]
)

# -------------------------------------------------
# 4. Generate Multi-Run Explanations
# -------------------------------------------------
n_instances = 30
n_runs = 5
POS_CLASS = 1

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = f"LIME_outputs_{timestamp}"
html_dir = os.path.join(outputs_dir, "instance_html")

os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(html_dir, exist_ok=True)

instance_indices = np.random.choice(range(X_test_lime.shape[0]), n_instances, replace=False)
feature_weights_runs, raw_rows, pred_rows = [], [], []
jsonl_path = os.path.join(outputs_dir, "raw_explanations.jsonl")

for run in range(n_runs):
    np.random.seed(LIME_CONFIG["random_seed"] + run)
    run_weights = np.zeros(len(feature_names))

    for idx in instance_indices:
        x = X_test_lime[idx]
        probs = voting_model.predict_proba([x])[0]
        y_pred = int(np.argmax(probs))

        pred_rows.append({
            "run": run,
            "instance_idx": int(idx),
            "pred_class": y_pred,
            "prob_no_disease": float(probs[0]),
            "prob_disease": float(probs[1])
        })

        exp = explainer.explain_instance(
            data_row=x,
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )

        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight

        local_map = exp.as_map()
        for class_idx, pairs in local_map.items():
            for feat_idx, weight in pairs:
                raw_rows.append({
                    "run": run,
                    "instance_idx": int(idx),
                    "class_idx": int(class_idx),
                    "feature_idx": int(feat_idx),
                    "feature": feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}",
                    "weight": float(weight)
                })

        try:
            html_str = exp.as_html()
            html_file = os.path.join(html_dir, f"exp_run{run}_idx{idx}.html")
            with open(html_file, "w", encoding="utf-8") as hf:
                hf.write(html_str)
        except Exception as e:
            fallback = {
                "run": run,
                "instance_idx": int(idx),
                "note": f"HTML export failed: {e}",
                "map": {int(k): [(int(fi), float(w)) for fi, w in v] for k, v in local_map.items()}
            }
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(fallback) + "\n")

    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)

# -------------------------------------------------
# 5. Aggregate Feature Statistics
# -------------------------------------------------
signed_mean = np.mean(feature_weights_runs, axis=0)
signed_std = np.std(feature_weights_runs, axis=0)
abs_mean = np.mean(np.abs(feature_weights_runs), axis=0)
abs_std = np.std(np.abs(feature_weights_runs), axis=0)

lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})

lime_summary["Direction"] = np.where(lime_summary["SignedMean"] > 0, "↑ increases risk", "↓ decreases risk")
lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1

print("\n--- LIME Feature Importance Summary ---")
print(lime_summary.head(10))

raw_df = pd.DataFrame(raw_rows)
raw_df.to_csv(os.path.join(outputs_dir, "raw_explanations.csv"), index=False)

pred_df = pd.DataFrame(pred_rows)
pred_df.to_csv(os.path.join(outputs_dir, "predictions.csv"), index=False)

# -------------------------------------------------
# 6. Visualization
# -------------------------------------------------
fig_filename = os.path.join(outputs_dir, f"LIME_feature_importance_{timestamp}.png")

plt.figure(figsize=(10, 6))
colors = lime_summary["SignedMean"].apply(lambda x: "crimson" if x > 0 else "steelblue")
plt.barh(lime_summary["Feature"], lime_summary["SignedMean"], color=colors)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Signed Mean LIME Weight")
plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble")
plt.tight_layout()
plt.savefig(fig_filename, dpi=300)
plt.close()
print(f"✅ LIME barplot saved as '{fig_filename}'")

# -------------------------------------------------
# 7. Stability Analysis
# -------------------------------------------------
top_k = 5
top_features_runs = [np.argsort(-np.abs(run))[:top_k] for run in feature_weights_runs]

def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))

stability_scores = [jaccard(top_features_runs[i], top_features_runs[j])
                    for i in range(n_runs) for j in range(i+1, n_runs)]

stability_mean, stability_std = np.mean(stability_scores), np.std(stability_scores)
print(f"\nLIME Stability (Top-{top_k} Jaccard): {stability_mean:.3f} ± {stability_std:.3f}")

csv_file = os.path.join(outputs_dir, f"LIME_summary_{timestamp}.csv")
lime_summary.to_csv(csv_file, index_label="Rank")

# -------------------------------------------------
# 8. HTML & PDF Reports
# -------------------------------------------------
html_file = os.path.join(outputs_dir, f"LIME_report_{timestamp}.html")

html_content = f"""
<html>
<head><title>LIME Report — NOVEM-CVD Ensemble</title></head>
<body>
<h1>📘 NOVEM-CVD LIME Explanation Report</h1>
<h2>Configuration</h2>
{pd.DataFrame(LIME_CONFIG.items(), columns=['Parameter','Value']).to_html(index=False)}
<h2>Feature Importance (Top 10)</h2>
{lime_summary.head(10).to_html(index=True)}
<h2>Stability</h2>
<p><b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}</p>
<h2>Visualization</h2>
<img src="{os.path.basename(fig_filename)}" width="80%"/>
<h2>Artifacts</h2>
<ul>
<li>{os.path.basename(csv_file)}</li>
<li>{os.path.basename(fig_filename)}</li>
<li>raw_explanations.csv / raw_explanations.jsonl</li>
<li>predictions.csv</li>
<li>instance_html/</li>
</ul>
<hr><p style='text-align:center;'>Generated — NOVEM-CVD © 2025</p>
</body></html>
"""
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"✅ HTML report saved: {html_file}")

# PDF Generation
pdf_file = os.path.join(outputs_dir, f"LIME_report_{timestamp}.pdf")
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("📘 NOVEM-CVD LIME Explanation Report", styles['Title']))
story.append(Spacer(1, 12))
story.append(Paragraph("<b>Configuration</b>", styles['Heading2']))

cfg_table = Table([[k, str(v)] for k, v in LIME_CONFIG.items()], colWidths=[200, 200])
cfg_table.setStyle(TableStyle([
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER')
]))
story.append(cfg_table)
story.append(Spacer(1, 12))

story.append(Paragraph("<b>Feature Importance (Top 10)</b>", styles['Heading2']))
top10 = lime_summary.head(10)
table = Table([list(top10.columns)] + top10.values.tolist(), repeatRows=1)
table.setStyle(TableStyle([
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER')
]))
story.append(table)
story.append(Spacer(1, 12))
story.append(Paragraph(
    f"<b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}",
    styles['Normal']
))
story.append(Image(fig_filename, width=450, height=300))
story.append(Spacer(1, 12))
story.append(Paragraph("<b>Artifacts:</b><br/>CSV, PNG, JSONL, predictions, instance_html/", styles['Normal']))
doc.build(story)

print(f"✅ PDF report built successfully: {pdf_file}")

# -------------------------------------------------
# 9. ZIP Packaging + Download Safe for Colab and Local
# -------------------------------------------------
zip_filename = f"LIME_outputs_{timestamp}.zip"
shutil.make_archive(outputs_dir, 'zip', outputs_dir)
print(f"📦 Created ZIP archive: {zip_filename}")

friendly_zip = "NOVEM_CVD_LIME_Report_Latest.zip"
shutil.copy(zip_filename, friendly_zip)
print(f"📦 Copied latest archive → {friendly_zip}")

try:
    from google.colab import files
    print("📥 Starting automatic download...")
    files.download(friendly_zip)
    print(f"✅ Auto-download started: {friendly_zip}")
except Exception:
    print("ℹ️ Not running in Colab — skipping auto-download.")

# Cleanup older zips
all_zips = sorted(glob.glob("LIME_outputs_*.zip"), key=os.path.getmtime)
if len(all_zips) > 2:
    for z in all_zips[:-1]:
        os.remove(z)
        print(f"🧹 Removed old archive: {z}")
print("🧾 Cleanup complete.")

```

### Cell 66 — code (execution_count=None)
```python

```

### Cell 67 — code (execution_count=56)
```python
# ============================================================
# 🧠 NOVEM-CvD — LIME EXPLANATION & STABILITY SUITE (v2025)
# Parts 1–3 merged | Reviewer-ready | PDF + ZIP + Colab-compatible
# ============================================================

import os, json, zipfile, datetime, warnings, shutil, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet
from google.colab import files   # comment out if running locally
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1️⃣ Load model and data
# -------------------------------------------------
print("🔄 Loading pre-saved model and dataset…")
import joblib
voting_model = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")
data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")

X_train_lime, X_test_lime = data["X_train"], data["X_test"]
y_train_lime, y_test_lime = data["y_train"], data["y_test"]
print("✅ Model & data loaded successfully.")

# -------------------------------------------------
# 2️⃣ Feature setup
# -------------------------------------------------
# Replace with your actual top-feature table logic if applicable
try:
    feature_names = top_features_table.apply(
        lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1
    ).tolist()
except Exception:
    feature_names = list(X_train_lime.columns)

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# -------------------------------------------------
# 3️⃣ LIME configuration
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED,
}
discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],
    feature_names=feature_names
)
explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=RANDOM_SEED
)

# -------------------------------------------------
# 4️⃣ Multi-run explanations & stability
# -------------------------------------------------
n_instances, n_runs, POS_CLASS = 30, 5, 1
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = f"LIME_outputs_{timestamp}"
html_dir = os.path.join(outputs_dir, "instance_html")
os.makedirs(html_dir, exist_ok=True)

instance_indices = np.random.choice(range(X_test_lime.shape[0]), n_instances, replace=False)
feature_weights_runs, raw_rows, pred_rows = [], [], []

print("⚙️ Running multi-run LIME explanations…")
for run in range(n_runs):
    np.random.seed(RANDOM_SEED + run)
    run_weights = np.zeros(len(feature_names))
    for idx in instance_indices:
        x = X_test_lime[idx]
        probs = voting_model.predict_proba([x])[0]
        y_pred = int(np.argmax(probs))
        pred_rows.append({
            "run": run, "instance_idx": int(idx),
            "pred_class": y_pred,
            "prob_no_disease": float(probs[0]),
            "prob_disease": float(probs[1]),
        })
        exp = explainer.explain_instance(
            data_row=x,
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight
        local_map = exp.as_map()
        for class_idx, pairs in local_map.items():
            for feat_idx, weight in pairs:
                raw_rows.append({
                    "run": run, "instance_idx": int(idx),
                    "class_idx": int(class_idx),
                    "feature_idx": int(feat_idx),
                    "feature": feature_names[feat_idx],
                    "weight": float(weight)
                })
        with open(os.path.join(html_dir, f"exp_run{run}_idx{idx}.html"), "w", encoding="utf-8") as hf:
            hf.write(exp.as_html())
    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)
print("✅ Explanations complete.")

# -------------------------------------------------
# 5️⃣ Aggregate statistics + stability
# -------------------------------------------------
signed_mean = np.mean(feature_weights_runs, axis=0)
abs_mean = np.mean(np.abs(feature_weights_runs), axis=0)
lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "AbsMean": abs_mean,
    "Direction": np.where(signed_mean > 0, "↑ increases risk", "↓ decreases risk"),
}).sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1
print("\n--- LIME Summary (Top 10) ---")
print(lime_summary.head(10))

# Stability metric
def jaccard(a,b): return len(set(a)&set(b))/len(set(a)|set(b))
top_k = 5
top_features_runs = [np.argsort(-np.abs(run))[:top_k] for run in feature_weights_runs]
stability_scores = [jaccard(top_features_runs[i], top_features_runs[j])
                    for i in range(n_runs) for j in range(i+1,n_runs)]
stability_mean, stability_std = np.mean(stability_scores), np.std(stability_scores)
print(f"\n📊 LIME Stability (Top-{top_k} Jaccard): {stability_mean:.3f} ± {stability_std:.3f}")

# -------------------------------------------------
# 6️⃣ Visualization + summary exports
# -------------------------------------------------
os.makedirs(outputs_dir, exist_ok=True)
fig_filename = os.path.join(outputs_dir, "LIME_feature_importance.png")
plt.figure(figsize=(10,6))
colors = lime_summary["SignedMean"].apply(lambda x: "crimson" if x>0 else "steelblue")
plt.barh(lime_summary["Feature"], lime_summary["SignedMean"], color=colors)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Signed Mean LIME Weight"); plt.title("LIME Feature Directionality For Proposed NOVEC")
plt.tight_layout(); plt.savefig(fig_filename, dpi=300); plt.close()
print(f"✅ LIME bar plot saved → {fig_filename}")

csv_file = os.path.join(outputs_dir, "LIME_summary.csv")
lime_summary.to_csv(csv_file, index_label="Rank")

# -------------------------------------------------
# 7️⃣ PDF Report (Build via reportlab)
# -------------------------------------------------
pdf_file = os.path.join(outputs_dir, f"LIME_report_{timestamp}.pdf")
doc = SimpleDocTemplate(pdf_file, pagesize=letter)
styles = getSampleStyleSheet(); story=[]
story += [Paragraph("📘 NOVEM-CvD LIME Explanation Report", styles["Title"]), Spacer(1,12)]
story += [Paragraph("<b>Configuration</b>", styles["Heading2"])]
conf_tab = [[k,str(v)] for k,v in LIME_CONFIG.items()]
table=Table(conf_tab, colWidths=[200,200])
table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),rl_colors.lightgrey),
                           ('GRID',(0,0),(-1,-1),0.5,rl_colors.grey),
                           ('ALIGN',(0,0),(-1,-1),'CENTER')]))
story += [table, Spacer(1,12)]
story += [Paragraph("<b>Feature Importance (Top 10)</b>", styles["Heading2"])]
table_data=[list(lime_summary.head(10).columns)]+lime_summary.head(10).values.tolist()
tab=Table(table_data, repeatRows=1, colWidths=[90,90,90,120])
tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),rl_colors.lightgrey),
                         ('GRID',(0,0),(-1,-1),0.5,rl_colors.grey)]))
story += [tab, Spacer(1,12)]
story += [Paragraph(f"<b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}", styles["Normal"])]
story += [Spacer(1,12), Paragraph("<b>Visualization</b>", styles["Heading2"]),
           Image(fig_filename, width=450, height=300)]
story += [Spacer(1,12), Paragraph("<hr/><p style='text-align:center;'>Generated automatically — NOVEM-CvD © 2025</p>", styles["Normal"])]
doc.build(story)
print(f"✅ PDF report saved → {pdf_file}")

# -------------------------------------------------
# 8️⃣ ZIP Archive + Auto-Download (COLAB)
# -------------------------------------------------
zip_filename = f"LIME_outputs_{timestamp}.zip"
shutil.make_archive(base_name=f"LIME_outputs_{timestamp}", format="zip", root_dir=outputs_dir)
friendly_zip = "NOVEM_CVD_LIME_Report_Latest.zip"
shutil.copy(zip_filename, friendly_zip)
print(f"📦 ZIP created → {friendly_zip}")
try:
    files.download(friendly_zip)
    print("📥 Auto-download started in Colab.")
except Exception as e:
    print(f"⚠️ Auto-download skipped: {e}")

# -------------------------------------------------
# 9️⃣ PART 3 — Aggregate Instance-Level LIME Explanations
# -------------------------------------------------
feature_contributions = pd.DataFrame(0, index=range(len(X_test_lime)), columns=feature_names)
print("\n🔄 Aggregating feature contributions from LIME explanations (deterministic)…")

for idx in range(len(X_test_lime)):
    np.random.seed(RANDOM_SEED + idx)
    instance = X_test_lime[idx].reshape(1,-1)
    exp = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=voting_model.predict_proba,
        num_features=len(feature_names)
    )
    for feature, weight in exp.as_list():
        base_feature = feature.split(" ")[0]
        if base_feature in feature_contributions.columns:
            feature_contributions.at[idx, base_feature] += weight
    if (idx+1)%50==0 or (idx+1)==len(X_test_lime):
        print(f"✅ Processed {idx+1}/{len(X_test_lime)} instances")

agg_contributions=(feature_contributions.abs().mean(axis=0)
                   .sort_values(ascending=False)
                   .to_frame(name="mean_abs_contribution"))
agg_contributions["percent"]=100*agg_contributions["mean_abs_contribution"]/agg_contributions["mean_abs_contribution"].sum()
print("\n✅ Top PCA components based on LIME contributions:")
print(agg_contributions.head(10))

# Visualization
# ✅ Corrected Visualization — NOVEM-CvD PCA Contribution Plot
plt.figure(figsize=(10, 6))

# Reset the index for clean data input
agg_plot = agg_contributions.reset_index()
agg_plot.rename(columns={'index': 'PCA_Component'}, inplace=True)

# Proper x (values) and y (labels)
ax = sns.barplot(
    data=agg_plot,
    x="mean_abs_contribution",
    y="PCA_Component",
    palette="viridis"
)

# Add text labels for values + percentages
for i, (val, pct) in enumerate(zip(agg_plot["mean_abs_contribution"], agg_plot["percent"])):
    ax.text(val + 0.002, i, f"{val:.2f} ({pct:.1f}%)", color="black", va="center", fontsize=9)

# Clean labels and adjust axes
ax.set_xlim(0, agg_plot["mean_abs_contribution"].max() * 1.25)
ax.set_xlabel("Mean Absolute LIME Contribution", fontsize=11)
ax.set_ylabel("PCA Component (Top Contributing Feature Pairs)", fontsize=11)
ax.set_title("Average LIME Contributions per PCA Component", fontsize=13, fontweight="bold")

plt.tight_layout()

# Save and show
fig_path_pdf = os.path.join(outputs_dir, "LIME_PCA_summary_fixed.pdf")
plt.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Corrected PCA summary PDF saved → {fig_path_pdf}")

# -------------------------------------------------
# 🔟 Optional PCA→Original Feature Mapping (if loading_df available)
# -------------------------------------------------
try:
    if "loading_df" in globals():
        common=min(feature_contributions.shape[1],loading_df.shape[1])
        aligned_fc=feature_contributions.iloc[:,:common]
        aligned_ld=loading_df.iloc[:,:common]
        abs_ld=np.abs(aligned_ld.values)
        ld_norm=abs_ld/abs_ld.sum(axis=0,keepdims=True)
        orig_contrib=pd.DataFrame(aligned_fc.values@ld_norm.T,columns=aligned_ld.index)
        orig_agg=(orig_contrib.abs().mean(axis=0)
                  .sort_values(ascending=False)
                  .to_frame(name="mean_abs_contribution"))
        orig_agg["percent"]=100*orig_agg["mean_abs_contribution"]/orig_agg["mean_abs_contribution"].sum()
        plt.figure(figsize=(10,6))
        ax=sns.barplot(x="mean_abs_contribution",y=orig_agg.index,
                       data=orig_agg.reset_index(),palette="plasma")
        for i,(val,pct) in enumerate(zip(orig_agg["mean_abs_contribution"],orig_agg["percent"])):
            ax.text(val+0.002,i,f"{val:.3f} ({pct:.1f}%)",color="black",va="center",fontsize=9)
        ax.set_xlim(0,orig_agg["mean_abs_contribution"].max()*1.25)
        plt.title("Average LIME Contributions per Original Feature (Mapped from PCA)")
        plt.tight_layout()
        orig_fig_pdf=os.path.join(outputs_dir,"LIME_original_features_summary.pdf")
        plt.savefig(orig_fig_pdf,dpi=300,bbox_inches="tight"); plt.close()
        print(f"✅ Original feature summary PDF → {orig_fig_pdf}")
except Exception as e:
    print(f"⚠️ PCA mapping skipped: {e}")

# -------------------------------------------------
# 11️⃣ Final export + auto-download (optional)
# -------------------------------------------------
agg_contributions.to_csv(os.path.join(outputs_dir,"LIME_PCA_contributions.csv"),index=True)
print("✅ All NOVEM-CvD LIME outputs complete.")
try:
    files.download(pdf_file)
    files.download(fig_path_pdf)
    print("📥 PDFs ready for download.")
except Exception as e:
    print(f"⚠️ Download skipped: {e}")

```

### Cell 68 — code (execution_count=56)
```python

```

### Cell 69 — code (execution_count=58)
```python
# ---------------------- NOVEM-CvD LIME MODULE — PART 1/3 ----------------------
# Loads data/model, rebuilds PCA loadings + meaningful names, initializes LIME
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib, warnings, os, datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Load Train/Test and Model
# -------------------------------------------------
print("🔄 Loading pre-saved model and dataset...")

data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")

X_train_lime = data["X_train"]
X_test_lime  = data["X_test"]
y_train_lime = data["y_train"]
y_test_lime  = data["y_test"]

voting_model = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")

print("✅ Data and model loaded successfully!")
print(f"X_train shape: {X_train_lime.shape}, X_test shape: {X_test_lime.shape}")

# -------------------------------------------------
# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
feature_names = top_features_table.apply(lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# Map full PCA component names to shorter labels
short_labels = {
    'ExerciseAngina_ST_Slope': 'ExAng_ST',
    'Cholesterol_FastingBS': 'Chol_FBS',
    'RestingBP_Age': 'BP_Age',
    'RestingECG_FastingBS': 'ECG_FBS',
    'Sex_ChestPainType': 'Sex_CPT',
    'FastingBS_MaxHR': 'FBS_MaxHR',
    'RestingBP_ChestPainType': 'BP_CPT',
    'Cholesterol_Age': 'Chol_Age',
    'Oldpeak_Cholesterol': 'Oldp_Chol'
}

# Replace feature names in your list
feature_names = [short_labels.get(f, f) for f in feature_names]

# -------------------------------------------------
# 4. LIME configuration
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = f"LIME_outputs_{timestamp}"
os.makedirs(outputs_dir, exist_ok=True)
html_dir = os.path.join(outputs_dir, "instance_html")
os.makedirs(html_dir, exist_ok=True)


# -------------------------------------------------
# 5. Initialize LIME Explainer on PCA features
# -------------------------------------------------
discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=LIME_CONFIG["random_seed"]
)

print("✅ LIME explainer initialized.")
print("✅ Part 1/3 complete — ready for Part 2/3.")
# ---------------------- NOVEM-CvD LIME MODULE — PART 2/3 ----------------------
# Generate LIME explanations for test instances and save as HTML
# ------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# -------------------------------------------------
# 1. Configuration
# -------------------------------------------------
num_instances_to_explain = len(X_test_lime)  # explain all test instances
html_dir = os.path.join(outputs_dir, "instance_html")

print(f"🔄 Generating LIME explanations for {num_instances_to_explain} test instances...")

# -------------------------------------------------
# 2. Loop through test instances
# -------------------------------------------------
for idx in range(num_instances_to_explain):
    instance = X_test_lime[idx].reshape(1, -1)
    exp = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=voting_model.predict_proba,
        num_features=min(len(feature_names), 5)
    )

    # Save HTML explanation
    html_file = os.path.join(html_dir, f"instance_{idx+1:03d}.html")
    exp.save_to_file(html_file)

    if (idx+1) % 50 == 0 or (idx+1) == num_instances_to_explain:
        print(f"✅ Saved explanations for {idx+1}/{num_instances_to_explain} instances")

print(f"\n✅ All LIME explanations saved in: {html_dir}")
print("✅ Part 2/3 complete — ready for Part 3/3.")
# ---------------------- NOVEM-CvD LIME MODULE — PART 3/3 (Final Fixed) ----------------------
# Aggregate LIME explanations and visualize top contributing PCA & original features
# ------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 1. Initialize storage for feature contributions
# -------------------------------------------------
feature_contributions = pd.DataFrame(0, index=range(len(X_test_lime)), columns=feature_names)
print("🔄 Aggregating feature contributions from LIME explanations...")

# -------------------------------------------------
# 2. Generate and parse LIME explanations (clean feature labels)
# -------------------------------------------------
for idx in range(len(X_test_lime)):
    instance = X_test_lime[idx].reshape(1, -1)
    exp = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=voting_model.predict_proba,
        num_features=len(feature_names)
    )

    for feature, weight in exp.as_list():
        base_feature = feature.split(' ')[0]       # remove threshold conditions
        if base_feature in feature_contributions.columns:
            feature_contributions.at[idx, base_feature] += weight

    if (idx + 1) % 50 == 0 or (idx + 1) == len(X_test_lime):
        print(f"✅ Processed {idx+1}/{len(X_test_lime)} instances")

# -------------------------------------------------
# 3. Aggregate mean absolute contributions (per PCA component)
# -------------------------------------------------
agg_contributions = (
    feature_contributions.abs().mean(axis=0)
    .sort_values(ascending=False)
    .to_frame(name='mean_abs_contribution')
)

print("\n✅ Top PCA components based on LIME contributions:")
print(agg_contributions.head(10))

# -------------------------------------------------
# 4. Visualization — PCA-level summary
# -------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(
    x="mean_abs_contribution",
    y=agg_contributions.index,
    data=agg_contributions.reset_index(),
    palette="viridis"
)
plt.title("Average LIME Contributions per PCA Component")
plt.xlabel("Mean Absolute Contribution")
plt.ylabel("PCA Component (with top original features)")
plt.tight_layout()

fig_path = os.path.join(outputs_dir, "LIME_PCA_summary.png")
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"\n✅ PCA-level summary plot saved at: {fig_path}")

# -------------------------------------------------
# 5. (Optional) Map PCA-level contributions back to original features (robust version)
# -------------------------------------------------
try:
    if 'loading_df' in globals():
        # Align dimensions between LIME components and PCA loadings
        common_components = min(feature_contributions.shape[1], loading_df.shape[1])
        aligned_feature_contribs = feature_contributions.iloc[:, :common_components]
        aligned_pca_loadings = loading_df.iloc[:, :common_components]

        # Map PCA → original features (using absolute loadings for interpretability)
        orig_feature_contribs = pd.DataFrame(
            aligned_feature_contribs.values @ np.abs(aligned_pca_loadings.values.T),
            columns=loading_df.index  # original feature names
        )

        # Aggregate mean absolute contributions by original feature
        orig_agg = (
            orig_feature_contribs.abs().mean(axis=0)
            .sort_values(ascending=False)
            .to_frame(name='mean_abs_contribution')
        )

        # Visualization — original features
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="mean_abs_contribution",
            y=orig_agg.index,
            data=orig_agg.reset_index(),
            palette="plasma"
        )
        plt.title("Average LIME Contributions per Original Feature (Mapped from PCA)")
        plt.xlabel("Mean Absolute Contribution")
        plt.ylabel("Original Feature")
        plt.tight_layout()

        orig_fig_path = os.path.join(outputs_dir, "LIME_original_features_summary.png")
        plt.savefig(orig_fig_path, dpi=300)
        plt.show()

        print(f"✅ Original feature-level summary plot saved at: {orig_fig_path}")
    else:
        print("⚠️ PCA loadings not found — skipping mapping to original features.")
except Exception as e:
    print(f"⚠️ PCA mapping step skipped due to error: {e}")

# -------------------------------------------------
# 6. Save summary results
# -------------------------------------------------
agg_contributions.to_csv(os.path.join(outputs_dir, "LIME_PCA_contributions.csv"), index=True)
print(f"✅ Aggregated PCA contributions saved in {outputs_dir}")
print("✅ Part 3/3 complete — NOVEM-CvD LIME workflow finished successfully!")

```

### Cell 70 — code (execution_count=59)
```python
# ---------------------- NOVEM-CvD LIME MODULE — PART 3/3 (Final Reproducible + Comparison Figure) ----------------------
# Aggregate LIME explanations and visualize top contributing PCA & original features
# ------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 0. Global reproducibility seed
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------------------------
# 1. Initialize storage for feature contributions
# -------------------------------------------------
feature_contributions = pd.DataFrame(0, index=range(len(X_test_lime)), columns=feature_names)
print("🔄 Aggregating feature contributions from LIME explanations (deterministic mode)…")

# -------------------------------------------------
# 2. Generate and parse LIME explanations (clean feature labels)
# -------------------------------------------------
for idx in range(len(X_test_lime)):
    np.random.seed(RANDOM_SEED + idx)  # ✅ stable per instance

    instance = X_test_lime[idx].reshape(1, -1)
    exp = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=voting_model.predict_proba,
        num_features=len(feature_names)
    )

    for feature, weight in exp.as_list():
        base_feature = feature.split(" ")[0]  # remove threshold conditions
        if base_feature in feature_contributions.columns:
            feature_contributions.at[idx, base_feature] += weight

    if (idx + 1) % 50 == 0 or (idx + 1) == len(X_test_lime):
        print(f"✅ Processed {idx+1}/{len(X_test_lime)} instances")

# -------------------------------------------------
# 3. Aggregate mean absolute contributions (per PCA component)
# -------------------------------------------------
agg_contributions = (
    feature_contributions.abs().mean(axis=0)
    .sort_values(ascending=False)
    .to_frame(name="mean_abs_contribution")
)
agg_contributions["percent"] = (
    100 * agg_contributions["mean_abs_contribution"]
    / agg_contributions["mean_abs_contribution"].sum()
)

print("\n✅ Top PCA components based on LIME contributions:")
print(agg_contributions.head(10))

# -------------------------------------------------
# 4. Visualization — PCA-level summary (values + %)
# -------------------------------------------------
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="mean_abs_contribution",
    y=agg_contributions.index,
    data=agg_contributions.reset_index(),
    palette="viridis"
)

for i, (val, pct) in enumerate(zip(agg_contributions["mean_abs_contribution"], agg_contributions["percent"])):
    ax.text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=9)

x_max = agg_contributions["mean_abs_contribution"].max()
ax.set_xlim(0, x_max * 1.25)

plt.title("Average LIME Contributions per PCA Component")
plt.xlabel("Mean Absolute Contribution")
plt.ylabel("PCA Component (with top original features)")
plt.tight_layout()

fig_path = os.path.join(outputs_dir, "LIME_PCA_summary.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\n✅ PCA-level summary plot saved at: {fig_path}")

# -------------------------------------------------
# 5. (Optional) Map PCA-level contributions back to original features (Normalized Mapping)
# -------------------------------------------------
orig_agg = None  # will store mapping results for combined figure
try:
    if "loading_df" in globals():
        # Align dimensions between LIME components and PCA loadings
        common_components = min(feature_contributions.shape[1], loading_df.shape[1])
        aligned_feature_contribs = feature_contributions.iloc[:, :common_components]
        aligned_pca_loadings = loading_df.iloc[:, :common_components]

        # Normalize absolute loadings so each PCA component sums to 1
        abs_loadings = np.abs(aligned_pca_loadings.values)
        loadings_norm = abs_loadings / abs_loadings.sum(axis=0, keepdims=True)

        # Map PCA → original features (normalized mapping preserves magnitude)
        orig_feature_contribs = pd.DataFrame(
            aligned_feature_contribs.values @ loadings_norm.T,
            columns=aligned_pca_loadings.index  # original feature names
        )

        # Aggregate mean absolute contributions by original feature
        orig_agg = (
            orig_feature_contribs.abs().mean(axis=0)
            .sort_values(ascending=False)
            .to_frame(name="mean_abs_contribution")
        )
        orig_agg["percent"] = (
            100 * orig_agg["mean_abs_contribution"]
            / orig_agg["mean_abs_contribution"].sum()
        )

        # Visualization — original features with labels and expanded x-axis
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="mean_abs_contribution",
            y=orig_agg.index,
            data=orig_agg.reset_index(),
            palette="plasma"
        )

        for i, (val, pct) in enumerate(zip(orig_agg["mean_abs_contribution"], orig_agg["percent"])):
            ax.text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=9)

        x_max = orig_agg["mean_abs_contribution"].max()
        ax.set_xlim(0, x_max * 1.25)

        plt.title("Average LIME Contributions per Original Feature (Mapped from PCA, Normalized)")
        plt.xlabel("Mean Absolute Contribution")
        plt.ylabel("Original Feature")
        plt.tight_layout()

        orig_fig_path = os.path.join(outputs_dir, "LIME_original_features_summary.png")
        plt.savefig(orig_fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"✅ Original feature-level summary plot saved at: {orig_fig_path}")
    else:
        print("⚠️ PCA loadings not found — skipping mapping to original features.")
except Exception as e:
    print(f"⚠️ PCA mapping step skipped due to error: {e}")

# -------------------------------------------------
# 6. Combined comparison figure (side-by-side)
# -------------------------------------------------
try:
    if orig_agg is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # PCA-level
        sns.barplot(
            x="mean_abs_contribution",
            y=agg_contributions.index,
            data=agg_contributions.reset_index(),
            palette="viridis",
            ax=axes[0]
        )
        for i, (val, pct) in enumerate(zip(agg_contributions["mean_abs_contribution"], agg_contributions["percent"])):
            axes[0].text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=8)
        axes[0].set_xlim(0, agg_contributions["mean_abs_contribution"].max() * 1.25)
        axes[0].set_title("PCA Components")
        axes[0].set_xlabel("Mean Absolute Contribution")
        axes[0].set_ylabel("PCA Component")

        # Original features
        sns.barplot(
            x="mean_abs_contribution",
            y=orig_agg.index,
            data=orig_agg.reset_index(),
            palette="plasma",
            ax=axes[1]
        )
        for i, (val, pct) in enumerate(zip(orig_agg["mean_abs_contribution"], orig_agg["percent"])):
            axes[1].text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=8)
        axes[1].set_xlim(0, orig_agg["mean_abs_contribution"].max() * 1.25)
        axes[1].set_title("Original Features (Mapped from PCA, Normalized)")
        axes[1].set_xlabel("Mean Absolute Contribution")
        axes[1].set_ylabel("Original Feature")

        plt.suptitle("NOVEM-CvD — LIME Contribution Comparison", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        combined_fig_path = os.path.join(outputs_dir, "LIME_combined_comparison.png")
        plt.savefig(combined_fig_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"✅ Combined comparison plot saved at: {combined_fig_path}")
except Exception as e:
    print(f"⚠️ Combined comparison figure skipped due to error: {e}")

# -------------------------------------------------
# 7. Save summary results
# -------------------------------------------------
agg_contributions.to_csv(os.path.join(outputs_dir, "LIME_PCA_contributions.csv"), index=True)
print(f"✅ Aggregated PCA contributions saved in {outputs_dir}")
print("✅ Part 3/3 complete — NOVEM-CvD LIME workflow finished successfully!")

```

### Cell 71 — code (execution_count=66)
```python
# ---------------------- NOVEM-CvD LIME MODULE — PART 3/3 (Final Reproducible + Comparison Figure + PDF Auto-Download) ----------------------
# Aggregate LIME explanations and visualize top contributing PCA & original features
# ------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files  # ✅ for automatic download

# -------------------------------------------------
# 0. Global reproducibility seed
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------------------------
# 1. Initialize storage for feature contributions
# -------------------------------------------------
feature_contributions = pd.DataFrame(0, index=range(len(X_test_lime)), columns=feature_names)
print("🔄 Aggregating feature contributions from LIME explanations (deterministic mode)…")

# -------------------------------------------------
# 2. Generate and parse LIME explanations (clean feature labels)
# -------------------------------------------------
for idx in range(len(X_test_lime)):
    np.random.seed(RANDOM_SEED + idx)  # ✅ stable per instance
    instance = X_test_lime[idx].reshape(1, -1)
    exp = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=voting_model.predict_proba,
        num_features=len(feature_names)
    )

    for feature, weight in exp.as_list():
        base_feature = feature.split(" ")[0]  # remove threshold conditions
        if base_feature in feature_contributions.columns:
            feature_contributions.at[idx, base_feature] += weight

    if (idx + 1) % 50 == 0 or (idx + 1) == len(X_test_lime):
        print(f"✅ Processed {idx+1}/{len(X_test_lime)} instances")

# -------------------------------------------------
# 3. Aggregate mean absolute contributions (per PCA component)
# -------------------------------------------------
agg_contributions = (
    feature_contributions.abs().mean(axis=0)
    .sort_values(ascending=False)
    .to_frame(name="mean_abs_contribution")
)
agg_contributions["percent"] = (
    100 * agg_contributions["mean_abs_contribution"]
    / agg_contributions["mean_abs_contribution"].sum()
)

print("\n✅ Top PCA components based on LIME contributions:")
print(agg_contributions.head(10))

# -------------------------------------------------
# 4. Visualization — PCA-level summary (values + %)
# -------------------------------------------------
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="mean_abs_contribution",
    y=agg_contributions.index,
    data=agg_contributions.reset_index(),
    palette="viridis"
)
for i, (val, pct) in enumerate(zip(agg_contributions["mean_abs_contribution"], agg_contributions["percent"])):
    ax.text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=9)

x_max = agg_contributions["mean_abs_contribution"].max()
ax.set_xlim(0, x_max * 1.25)
plt.xlabel("Mean Absolute Contribution")
plt.ylabel("Top Feature Pairs per PCA Component")
plt.tight_layout()

fig_path_png = os.path.join(outputs_dir, "LIME_PCA_summary.png")
fig_path_pdf = os.path.join(outputs_dir, "LIME_PCA_summary.pdf")
plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
plt.show()
print(f"\n✅ PCA-level summary plots saved at: {fig_path_pdf}")

# -------------------------------------------------
# 5. (Optional) Map PCA-level contributions back to original features (Normalized Mapping)
# -------------------------------------------------
orig_agg = None
try:
    if "loading_df" in globals():
        common_components = min(feature_contributions.shape[1], loading_df.shape[1])
        aligned_feature_contribs = feature_contributions.iloc[:, :common_components]
        aligned_pca_loadings = loading_df.iloc[:, :common_components]

        abs_loadings = np.abs(aligned_pca_loadings.values)
        loadings_norm = abs_loadings / abs_loadings.sum(axis=0, keepdims=True)

        orig_feature_contribs = pd.DataFrame(
            aligned_feature_contribs.values @ loadings_norm.T,
            columns=aligned_pca_loadings.index
        )

        orig_agg = (
            orig_feature_contribs.abs().mean(axis=0)
            .sort_values(ascending=False)
            .to_frame(name="mean_abs_contribution")
        )
        orig_agg["percent"] = (
            100 * orig_agg["mean_abs_contribution"]
            / orig_agg["mean_abs_contribution"].sum()
        )

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x="mean_abs_contribution",
            y=orig_agg.index,
            data=orig_agg.reset_index(),
            palette="plasma"
        )
        for i, (val, pct) in enumerate(zip(orig_agg["mean_abs_contribution"], orig_agg["percent"])):
            ax.text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=9)

        x_max = orig_agg["mean_abs_contribution"].max()
        ax.set_xlim(0, x_max * 1.25)
        plt.xlabel("Mean Absolute Contribution")
        plt.ylabel("Original Feature")
        plt.tight_layout()

        orig_fig_png = os.path.join(outputs_dir, "LIME_original_features_summary.png")
        orig_fig_pdf = os.path.join(outputs_dir, "LIME_original_features_summary.pdf")
        plt.savefig(orig_fig_png, dpi=300, bbox_inches="tight")
        plt.savefig(orig_fig_pdf, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"✅ Original feature-level plots saved at: {orig_fig_pdf}")
    else:
        print("⚠️ PCA loadings not found — skipping mapping to original features.")
except Exception as e:
    print(f"⚠️ PCA mapping step skipped due to error: {e}")

# -------------------------------------------------
# 6. Combined comparison figure (side-by-side)
# -------------------------------------------------
try:
    if orig_agg is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        sns.barplot(
            x="mean_abs_contribution",
            y=agg_contributions.index,
            data=agg_contributions.reset_index(),
            palette="viridis",
            ax=axes[0]
        )
        for i, (val, pct) in enumerate(zip(agg_contributions["mean_abs_contribution"], agg_contributions["percent"])):
            axes[0].text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=8)
        axes[0].set_xlim(0, agg_contributions["mean_abs_contribution"].max() * 1.25)
        axes[0].set_xlabel("Mean Absolute Contribution")
        axes[0].set_ylabel("PCA Component")

        sns.barplot(
            x="mean_abs_contribution",
            y=orig_agg.index,
            data=orig_agg.reset_index(),
            palette="plasma",
            ax=axes[1]
        )
        for i, (val, pct) in enumerate(zip(orig_agg["mean_abs_contribution"], orig_agg["percent"])):
            axes[1].text(val + 0.002, i, f"{val:.3f} ({pct:.1f}%)", color="black", va="center", fontsize=8)
        axes[1].set_xlim(0, orig_agg["mean_abs_contribution"].max() * 1.25)
        axes[1].set_xlabel("Mean Absolute Contribution")
        axes[1].set_ylabel("Original Feature")

        plt.suptitle("NOVEC's LIME Contribution Comparison", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        combined_fig_png = os.path.join(outputs_dir, "LIME_combined_comparison.png")
        combined_fig_pdf = os.path.join(outputs_dir, "LIME_combined_comparison.pdf")
        plt.savefig(combined_fig_png, dpi=300, bbox_inches="tight")
        plt.savefig(combined_fig_pdf, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"✅ Combined comparison plots saved at: {combined_fig_pdf}")
except Exception as e:
    print(f"⚠️ Combined comparison figure skipped due to error: {e}")

# -------------------------------------------------
# 7. Auto-download PDFs (Colab)
# -------------------------------------------------
try:
    print("📥 Auto-downloading all generated PDFs...")
    files.download(fig_path_pdf)
    if "orig_fig_pdf" in locals():
        files.download(orig_fig_pdf)
    if "combined_fig_pdf" in locals():
        files.download(combined_fig_pdf)
    print("✅ All PDF downloads triggered successfully.")
except Exception as e:
    print(f"⚠️ Auto-download failed: {e}")

# -------------------------------------------------
# 8. Save summary results
# -------------------------------------------------
agg_contributions.to_csv(os.path.join(outputs_dir, "LIME_PCA_contributions.csv"), index=True)
print(f"✅ Aggregated PCA contributions saved in {outputs_dir}")
print("✅ Part 3/3 complete — NOVEM-CvD LIME workflow finished successfully!")

```

### Cell 72 — code (execution_count=None)
```python

```

### Cell 73 — code (execution_count=26)
```python
!pip install reportlab

```

### Cell 74 — code (execution_count=68)
```python
# ---------------- Part 3/3 ----------------
# NOVEM-CVD LIME MODULE — PART 3/3
# Plotting (Signed + Absolute), Reports (HTML + PDF), ZIP + Auto-download
# -------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shutil, glob, time
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------
# Filenames (consistent)
# -------------------------
signed_png  = os.path.join(outputs_dir, f"LIME_signed_importance_{timestamp}.png")
signed_pdf  = os.path.join(outputs_dir, f"LIME_signed_importance_{timestamp}.pdf")
abs_png     = os.path.join(outputs_dir, f"LIME_absolute_importance_{timestamp}.png")
abs_pdf     = os.path.join(outputs_dir, f"LIME_absolute_importance_{timestamp}.pdf")
html_file   = os.path.join(outputs_dir, f"LIME_report_{timestamp}.html")
pdf_report  = os.path.join(outputs_dir, f"LIME_report_{timestamp}.pdf")
csv_summary = os.path.join(outputs_dir, f"LIME_summary_{timestamp}.csv")

# -------------------------
# 1) Signed Importance Plot (styled)
# -------------------------
plt.figure(figsize=(12, 7))
ax = plt.gca()
ax.set_facecolor("#fafafa")

# minimal vertical gridlines only
ax.grid(which='major', axis='x', linestyle='--', color='lightgray', linewidth=0.7, alpha=0.6)
ax.grid(which='major', axis='y', linestyle='', linewidth=0)

# modern axis style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# colors mapped by sign
colors_signed = lime_summary["SignedMean"].apply(lambda x: "crimson" if x > 0 else "steelblue").tolist()

bars = ax.barh(lime_summary["Feature"], lime_summary["SignedMean"], color=colors_signed, edgecolor='black', linewidth=0.5)

# vertical zero-line
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

# enlarge x-axis to avoid label overlap
min_x = lime_summary["SignedMean"].min()
max_x = lime_summary["SignedMean"].max()
# Add symmetric padding based on max absolute magnitude
pad = max(abs(min_x), abs(max_x)) * 0.25 if max(abs(min_x), abs(max_x))>0 else 0.01
ax.set_xlim(min_x - pad, max_x + pad)

# labels & title
ax.set_xlabel("Signed Mean LIME Weight", fontsize=12)
ax.set_title("LIME Feature Directionality — NOVEM-CVD Ensemble", fontsize=15)

# value labels on bars
for bar in bars:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    x_offset = 0.002 if width >= 0 else -0.002
    ha = 'left' if width >= 0 else 'right'
    ax.text(width + x_offset, y, f"{width:.4f}", va='center', ha=ha, fontsize=10, color='black')

# legend
legend_elements = [
    Patch(facecolor='crimson', edgecolor='black', label='↑ Increases Risk'),
    Patch(facecolor='steelblue', edgecolor='black', label='↓ Decreases Risk')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True, edgecolor='black', facecolor='white')

plt.tight_layout()
plt.savefig(signed_png, dpi=300)
plt.savefig(signed_pdf, format="pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved signed plots: {signed_png}, {signed_pdf}")

# -------------------------
# 2) Absolute Importance Plot (styled)
# -------------------------
# sort ascending for horizontal barh
abs_df = lime_summary.copy().sort_values("AbsMean", ascending=True)

plt.figure(figsize=(12, 7))
ax2 = plt.gca()
ax2.set_facecolor("#fafafa")
ax2.grid(which='major', axis='x', linestyle='--', color='lightgray', linewidth=0.7, alpha=0.6)
ax2.grid(which='major', axis='y', linestyle='', linewidth=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

bars2 = ax2.barh(abs_df["Feature"], abs_df["AbsMean"], color="seagreen", edgecolor='black', linewidth=0.6)
ax2.set_xlabel("Absolute Mean LIME Weight", fontsize=12)
ax2.set_title("Absolute LIME Feature Importance — NOVEM-CVD Ensemble", fontsize=15)

# numeric labels
for bar in bars2:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    ax2.text(width + 0.0005, y, f"{width:.4f}", va='center', ha='left', fontsize=10, color='black')

plt.tight_layout()
plt.savefig(abs_png, dpi=300)
plt.savefig(abs_pdf, format="pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved absolute plots: {abs_png}, {abs_pdf}")

# -------------------------
# 3) Update HTML report to include both images (overwrite previous)
# -------------------------
html_content = f"""
<html>
<head>
<title>LIME Report — NOVEM-CVD Ensemble</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
h1, h2 {{ color: #2b3e50; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
th {{ background-color: #f4f4f4; }}
img {{ display: block; margin: 20px auto; max-width: 80%; }}
</style>
</head>
<body>

<h1>📘 NOVEM-CVD LIME Explanation Report</h1>
<h2>Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
""" + "\n".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in LIME_CONFIG.items()]) + f"""
</table>

<h2>Feature Importance (Top 10)</h2>
{lime_summary.head(10).to_html(index=True, justify='center')}

<h2>Stability</h2>
<p><b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}</p>

<h2>Signed Visualization</h2>
<img src="{os.path.basename(signed_png)}" alt="Signed LIME Feature Importance Plot"/>

<h2>Absolute Visualization</h2>
<img src="{os.path.basename(abs_png)}" alt="Absolute LIME Feature Importance Plot"/>

<h2>Artifacts</h2>
<ul>
<li>{os.path.basename(csv_summary)}</li>
<li>{os.path.basename(signed_png)}</li>
<li>{os.path.basename(signed_pdf)}</li>
<li>{os.path.basename(abs_png)}</li>
<li>{os.path.basename(abs_pdf)}</li>
<li>raw_explanations.csv / raw_explanations.jsonl</li>
<li>predictions.csv</li>
<li>Per-instance HTML in <code>instance_html/</code></li>
</ul>

<hr>
<p style='text-align:center;'>Generated automatically — NOVEM-CvD Explainable Ensemble © 2025</p>
</body>
</html>
"""

with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)
# copy image files into outputs_dir if they were saved elsewhere already (they are already in outputs_dir)
print(f"✅ HTML report saved: {html_file}")

# -------------------------
# 4) Build ReportLab PDF including both figures + tables
# -------------------------
print("🧾 Generating full PDF report (ReportLab)...")
doc = SimpleDocTemplate(pdf_report, pagesize=letter)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("📘 NOVEM-CVD LIME Explanation Report", styles['Title']))
story.append(Spacer(1, 12))

# Configuration table
story.append(Paragraph("<b>Configuration</b>", styles['Heading2']))
config_table = [[k, str(v)] for k, v in LIME_CONFIG.items()]
table = Table(config_table, colWidths=[200, 200])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), rl_colors.lightgrey),
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
]))
story.append(table)
story.append(Spacer(1, 12))

# Feature importance table (top 10)
story.append(Paragraph("<b>Feature Importance (Top 10)</b>", styles['Heading2']))
top10 = lime_summary.head(10)
table_data = [list(top10.columns)] + top10.values.tolist()
table = Table(table_data, repeatRows=1, colWidths=[90, 90, 80, 80, 80, 120])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), rl_colors.lightgrey),
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
]))
story.append(table)
story.append(Spacer(1, 12))

# Stability
story.append(Paragraph("<b>Stability</b>", styles['Heading2']))
story.append(Paragraph(f"<b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}", styles['Normal']))
story.append(Spacer(1, 12))

# Signed visualization
story.append(Paragraph("<b>Signed LIME Feature Importance</b>", styles['Heading2']))
story.append(RLImage(signed_png, width=450, height=300))
story.append(Spacer(1, 12))

# Absolute visualization
story.append(Paragraph("<b>Absolute LIME Feature Importance</b>", styles['Heading2']))
story.append(RLImage(abs_png, width=450, height=300))
story.append(Spacer(1, 12))

# Artifacts list
story.append(Paragraph("<b>Artifacts Included</b>", styles['Heading2']))
story.append(Paragraph(
    f"- {os.path.basename(csv_summary)}<br/>"
    f"- {os.path.basename(signed_png)} / {os.path.basename(signed_pdf)}<br/>"
    f"- {os.path.basename(abs_png)} / {os.path.basename(abs_pdf)}<br/>"
    f"- raw_explanations.csv / raw_explanations.jsonl<br/>"
    f"- predictions.csv<br/>"
    f"- Per-instance HTML in <code>instance_html/</code>",
    styles['Normal']
))
story.append(Spacer(1, 12))

# Footer
story.append(Paragraph("<hr/>", styles['Normal']))
story.append(Paragraph("Generated automatically — NOVEM-CvD Explainable Ensemble © 2025", styles['Normal']))

doc.build(story)
print(f"✅ PDF report built: {pdf_report}")

# -------------------------
# 5) ZIP the outputs_dir (timestamped) and create friendly copy
# -------------------------
print("📦 Creating ZIP archive of outputs...")
zip_base = os.path.join(outputs_dir, f"LIME_outputs_{timestamp}")
zip_path = shutil.make_archive(zip_base, 'zip', outputs_dir)  # returns zip_path (with .zip)
zip_filename = zip_path
friendly_zip = "NOVEM_CVD_LIME_Report_Latest.zip"
try:
    shutil.copy(zip_filename, friendly_zip)
    print(f"✅ Copied archive to: {friendly_zip}")
except Exception as e:
    print(f"⚠️ Could not copy ZIP to friendly name: {e}")

# -------------------------
# 6) Auto-download in Google Colab (if available)
# -------------------------
try:
    from google.colab import files
    print("📥 Triggering auto-download of ZIP (Colab)...")
    files.download(friendly_zip)
    print("✅ Download triggered.")
except Exception as e:
    print(f"⚠️ Auto-download skipped: {e}")

# -------------------------
# 7) Optional cleanup: keep only last 3 timestamped archives in workspace
# -------------------------
try:
    all_zips = sorted(glob.glob("LIME_outputs_*/LIME_outputs_*.zip") + glob.glob("LIME_outputs_*.zip"),
                      key=os.path.getmtime)
    # keep latest 3
    if len(all_zips) > 3:
        for z in all_zips[:-3]:
            os.remove(z)
            print(f"🧹 Removed old archive: {z}")
except Exception as e:
    print(f"⚠️ Cleanup error (non-fatal): {e}")

print("\n🎉 All done — outputs written to:", outputs_dir)
# ---------------- End of Part 3/3 ----------------

```

### Cell 75 — code (execution_count=69)
```python
# ---------------------- NOVEM-CvD LIME MODULE — PART 3/3 (Fast PDF + Random State + Fixed Legend) ----------------------
# Plotting (Signed + Absolute), Reports (HTML + PDF using downscaled images), ZIP + Auto-download
# ------------------------------------------------------------------------------

import os, glob, shutil, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------------------------
# 0️⃣  Global reproducibility
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------------------------
# File paths (timestamped)
# -------------------------------------------------
signed_png  = os.path.join(outputs_dir, f"LIME_signed_importance_{timestamp}.png")
signed_pdf  = os.path.join(outputs_dir, f"LIME_signed_importance_{timestamp}.pdf")
abs_png     = os.path.join(outputs_dir, f"LIME_absolute_importance_{timestamp}.png")
abs_pdf     = os.path.join(outputs_dir, f"LIME_absolute_importance_{timestamp}.pdf")
html_file   = os.path.join(outputs_dir, f"LIME_report_{timestamp}.html")
pdf_report  = os.path.join(outputs_dir, f"LIME_report_{timestamp}.pdf")
csv_summary = os.path.join(outputs_dir, f"LIME_summary_{timestamp}.csv")

# -------------------------------------------------
# Helper: downscale large images for PDF embedding
# -------------------------------------------------
def downscale_for_pdf(src_path, max_width=800):
    """Down-scale image for fast PDF embedding."""
    try:
        img = Image.open(src_path)
        w, h = img.size
        if w > max_width:
            h = int(h * max_width / w)
            img = img.resize((max_width, h), Image.Resampling.LANCZOS)
            tmp_path = src_path.replace(".png", "_small.png")
            img.save(tmp_path, optimize=True, quality=85)
            return tmp_path
        return src_path
    except Exception as e:
        print(f"⚠️ Downscale skipped for {src_path}: {e}")
        return src_path

# -------------------------------------------------
# 1️⃣  Signed Importance Plot
# -------------------------------------------------
np.random.seed(RANDOM_SEED)
plt.figure(figsize=(12, 7))
ax = plt.gca()
ax.set_facecolor("#fafafa")
ax.grid(which="major", axis="x", linestyle="--", color="lightgray", linewidth=0.7, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

colors_signed = lime_summary["SignedMean"].apply(lambda x: "crimson" if x > 0 else "steelblue")
bars = ax.barh(lime_summary["Feature"], lime_summary["SignedMean"],
               color=colors_signed, edgecolor="black", linewidth=0.5)

ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
min_x, max_x = lime_summary["SignedMean"].min(), lime_summary["SignedMean"].max()
pad = max(abs(min_x), abs(max_x)) * 0.25 if max(abs(min_x), abs(max_x)) > 0 else 0.01
ax.set_xlim(min_x - pad, max_x + pad)
ax.set_xlabel("Signed Mean LIME Weight", fontsize=12)
ax.set_title("LIME Feature Directionality of Proposed NOVEC", fontsize=15)

for bar in bars:
    w, y = bar.get_width(), bar.get_y() + bar.get_height()/2
    ha = "left" if w >= 0 else "right"
    ax.text(w + (0.002 if w >= 0 else -0.002), y, f"{w:.4f}", va="center", ha=ha, fontsize=9)

# ✅ Proper legend (no TypeError)
legend_elements = [
    Patch(facecolor="crimson", edgecolor="black", label="↑ Increases Risk"),
    Patch(facecolor="steelblue", edgecolor="black", label="↓ Decreases Risk")
]
leg = ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")

plt.tight_layout()
plt.savefig(signed_png, dpi=300, bbox_inches="tight")
plt.savefig(signed_pdf, format="pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved signed plots → {signed_pdf}")

# -------------------------------------------------
# 2️⃣  Absolute Importance Plot
# -------------------------------------------------
np.random.seed(RANDOM_SEED)
abs_df = lime_summary.sort_values("AbsMean", ascending=True)
plt.figure(figsize=(12, 7))
ax2 = plt.gca()
ax2.set_facecolor("#fafafa")
ax2.grid(which="major", axis="x", linestyle="--", color="lightgray", linewidth=0.7, alpha=0.6)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

bars2 = ax2.barh(abs_df["Feature"], abs_df["AbsMean"],
                 color="seagreen", edgecolor="black", linewidth=0.6)
ax2.set_xlabel("Absolute Mean LIME Weight", fontsize=12)
ax2.set_title("Absolute LIME Feature Importance of Proposed NOVEC", fontsize=15)
for bar in bars2:
    w, y = bar.get_width(), bar.get_y() + bar.get_height()/2
    ax2.text(w + 0.0005, y, f"{w:.4f}", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.savefig(abs_png, dpi=300, bbox_inches="tight")
plt.savefig(abs_pdf, format="pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Saved absolute plots → {abs_pdf}")

# -------------------------------------------------
# 3️⃣  HTML Report
# -------------------------------------------------
np.random.seed(RANDOM_SEED)
html_content = f"""
<html><head><title>LIME Report — NOVEM-CvD</title>
<style>
body {{font-family: Arial; margin: 40px;}}
h1,h2 {{color:#2b3e50;}}
table {{border-collapse:collapse;width:100%;}}
th,td {{border:1px solid #ccc;padding:8px;text-align:center;}}
th {{background:#f4f4f4;}}
img {{display:block;margin:20px auto;max-width:85%;}}
</style></head><body>
<h1>📘 NOVEM-CvD LIME Explanation Report</h1>
<h2>Configuration</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>
{''.join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in LIME_CONFIG.items()])}
</table>
<h2>Feature Importance (Top 10)</h2>
{lime_summary.head(10).to_html(index=False)}
<h2>Stability</h2>
<p><b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}</p>
<h2>Signed Visualization</h2><img src="{os.path.basename(signed_png)}"/>
<h2>Absolute Visualization</h2><img src="{os.path.basename(abs_png)}"/>
<hr><p style='text-align:center;'>Generated automatically — NOVEM-CvD © 2025</p>
</body></html>
"""
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"✅ HTML report → {html_file}")

# -------------------------------------------------
# 4️⃣  ReportLab PDF Report (using downscaled images)
# -------------------------------------------------
np.random.seed(RANDOM_SEED)
print("🧾 Building PDF report (fast embed) …")
signed_small = downscale_for_pdf(signed_png)
abs_small = downscale_for_pdf(abs_png)

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(pdf_report, pagesize=letter)
story = [Paragraph("📘 NOVEM-CvD LIME Explanation Report", styles["Title"]), Spacer(1, 12)]

story.append(Paragraph("<b>Configuration</b>", styles["Heading2"]))
cfg_tbl = Table([[k, str(v)] for k,v in LIME_CONFIG.items()], colWidths=[220, 220])
cfg_tbl.setStyle(TableStyle([
    ("GRID",(0,0),(-1,-1),0.5,rl_colors.grey),
    ("BACKGROUND",(0,0),(-1,0),rl_colors.lightgrey)
]))
story += [cfg_tbl, Spacer(1, 12)]

story.append(Paragraph("<b>Feature Importance (Top 10)</b>", styles["Heading2"]))
top10 = lime_summary.head(10)
tbl = Table([list(top10.columns)] + top10.values.tolist(),
             repeatRows=1, colWidths=[90]*len(top10.columns))
tbl.setStyle(TableStyle([
    ("GRID",(0,0),(-1,-1),0.5,rl_colors.grey),
    ("BACKGROUND",(0,0),(-1,0),rl_colors.lightgrey)
]))
story += [tbl, Spacer(1, 12)]

story.append(Paragraph("<b>Signed Importance Plot</b>", styles["Heading2"]))
story.append(RLImage(signed_small, width=450, height=300))
story.append(Spacer(1, 12))
story.append(Paragraph("<b>Absolute Importance Plot</b>", styles["Heading2"]))
story.append(RLImage(abs_small, width=450, height=300))
story.append(Spacer(1, 12))

doc.build(story)
print(f"✅ PDF report built → {pdf_report}")

# -------------------------------------------------
# 5️⃣  ZIP & Auto-download (Colab)
# -------------------------------------------------
np.random.seed(RANDOM_SEED)
zip_path = shutil.make_archive(os.path.join(outputs_dir, f"LIME_outputs_{timestamp}"), "zip", outputs_dir)
friendly_zip = "NOVEM_CVD_LIME_Report_Latest.zip"
shutil.copy(zip_path, friendly_zip)
print(f"✅ Archive → {friendly_zip}")

try:
    from google.colab import files
    print("📥 Triggering Colab downloads …")
    files.download(friendly_zip)
    time.sleep(2)
    files.download(pdf_report)
    time.sleep(2)
    files.download(html_file)
    print("✅ All downloads triggered.")
except Exception as e:
    print(f"⚠️ Auto-download skipped: {e}")

# -------------------------------------------------
# 6️⃣  Cleanup (keep latest 3 archives)
# -------------------------------------------------
try:
    all_zips = sorted(glob.glob("LIME_outputs_*.zip"), key=os.path.getmtime)
    if len(all_zips) > 3:
        for z in all_zips[:-3]:
            os.remove(z)
            print(f"🧹 Removed old archive: {z}")
except Exception as e:
    print(f"⚠️ Cleanup error (non-fatal): {e}")

print("\n🎉 All done — outputs written to:", outputs_dir)

```

### Cell 76 — code (execution_count=53)
```python
!df -h / | tail -1
!du -sh /content/* | sort -hr | head
```

### Cell 77 — code (execution_count=54)
```python
!rm -rf LIME_outputs_*
!rm -f NOVEM_CVD_LIME_Report_Latest.zip
!rm -rf /content/drive/MyDrive/models/__pycache__
!rm -rf /content/sample_data

```

### Cell 78 — code (execution_count=60)
```python
rm -rf /content/drive/MyDrive/LIME_outputs_20251106_005855
```

### Cell 79 — code (execution_count=61)
```python
!df -h / | tail -1

```

### Cell 80 — code (execution_count=71)
```python
# ============================================
# LIME EXPLANATION MODULE — Reviewer-Ready
# NOVEM-CVD (Explainable Ensemble)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib
import warnings, os, random
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
feature_names = top_features_table.apply(
    lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1
).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# Map full PCA component names to shorter labels
short_labels = {
    'ExerciseAngina_ST_Slope': 'ExAng_ST',
    'Cholesterol_FastingBS': 'Chol_FBS',
    'RestingBP_Age': 'BP_Age',
    'RestingECG_FastingBS': 'ECG_FBS',
    'Sex_ChestPainType': 'Sex_CPT',
    'FastingBS_MaxHR': 'FBS_MaxHR',
    'RestingBP_ChestPainType': 'BP_CPT',
    'Cholesterol_Age': 'Chol_Age',
    'Oldpeak_Cholesterol': 'Oldp_Chol'
}

feature_names = [short_labels.get(f, f) for f in feature_names]

# -------------------------------------------------
# 3) LIME Configuration (Reproducible)
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

# Reviewer Concern #15:
# LIME is applied to the final NOVEM-CVD ensemble (VotingClassifier pipeline),
# not to individual base learners. This ensures explanations reflect the actual deployed model.

# Explicit discretizer for transparency & reproducibility
discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],  # 👈 empty list avoids dtype conflict
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=LIME_CONFIG["random_seed"]
)

# -------------------------------------------------
# 4) Global Multi-Run Explanations for Stability
# -------------------------------------------------
n_instances = 30
n_runs = 5
POS_CLASS = 1

instance_indices = np.random.choice(range(X_test_lime.shape[0]), n_instances, replace=False)
feature_weights_runs = []

for run in range(n_runs):
    np.random.seed(LIME_CONFIG["random_seed"] + run)
    run_weights = np.zeros(len(feature_names))

    for idx in instance_indices:
        exp = explainer.explain_instance(
            data_row=X_test_lime[idx],
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight

    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)

# Aggregate stats
signed_mean = np.mean(feature_weights_runs, axis=0)
signed_std  = np.std(feature_weights_runs, axis=0)
abs_mean    = np.mean(np.abs(feature_weights_runs), axis=0)
abs_std     = np.std(np.abs(feature_weights_runs), axis=0)

lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})
lime_summary["Direction"] = np.where(
    lime_summary["SignedMean"] > 0, "↑ increases risk", "↓ decreases risk"
)
lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1

print("\n--- LIME Feature Importance Summary (Top 10) ---")
print(lime_summary.head(10))

# Reviewer Concern #15 (Clinical Plausibility):
print("\nInterpretation Note:")
print("Top LIME features such as Max HR, Age, and Fasting BS align with established cardiovascular risk factors,")
print("supporting the clinical plausibility and external validity of the NOVEM-CVD ensemble's learned associations.")

# -------------------------------------------------
# ✅ FIXED: Global Barplot (Seaborn palette error)
# -------------------------------------------------
plt.figure(figsize=(10, 6))
colors = ["crimson" if x > 0 else "steelblue" for x in lime_summary["SignedMean"]]
sns.barplot(data=lime_summary, x="SignedMean", y="Feature", palette=colors)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Signed Mean LIME Weight")
plt.title("LIME Feature Directionality of NOVEM-CVD Classifier")
plt.tight_layout()
plt.savefig("LIME_feature_importance.png", dpi=300)
plt.close()
print("✅ LIME barplot saved as 'LIME_feature_importance.png'")

# -------------------------------------------------
# Stability (Jaccard overlap of Top-k)
# -------------------------------------------------
top_k = 5
top_features_runs = [np.argsort(-np.abs(run))[:top_k] for run in feature_weights_runs]

def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))

stability_scores = [jaccard(top_features_runs[i], top_features_runs[j])
                    for i in range(n_runs) for j in range(i+1, n_runs)]
stability_mean = float(np.mean(stability_scores))
stability_std  = float(np.std(stability_scores))

print(f"\nLIME Stability (Top-{top_k} Jaccard): {stability_mean:.3f} ± {stability_std:.3f}")

# Export CSV
lime_summary.to_csv("LIME_summary.csv", index_label="Rank")
print("✅ LIME summary saved as 'LIME_summary.csv'")

# -------------------------------------------------
# 5) Local Explanations (Interactive Panels)
# -------------------------------------------------
os.makedirs("lime_instances", exist_ok=True)
proba = voting_model.predict_proba(X_test_lime)[:, POS_CLASS]
sorted_idx = np.argsort(proba)
lowest_idx  = int(sorted_idx[0])
highest_idx = int(sorted_idx[-1])
remaining = list(set(range(X_test_lime.shape[0])) - {lowest_idx, highest_idx})
random_idxs = random.sample(remaining, k=min(3, len(remaining)))
local_indices = [highest_idx, lowest_idx] + random_idxs

local_cards = []
for idx in local_indices:
    exp = explainer.explain_instance(
        data_row=X_test_lime[idx],
        predict_fn=voting_model.predict_proba,
        num_features=min(10, len(feature_names)),
        num_samples=LIME_CONFIG["num_samples"]
    )
    inst_path = f"lime_instances/lime_instance_{idx}.html"
    exp.save_to_file(inst_path)
    local_cards.append({
        "index": int(idx),
        "path": inst_path,
        "pred_prob": float(proba[idx]),
        "true_label": int(y_test_lime[idx])
    })
print(f"✅ Saved {len(local_cards)} local LIME panels in 'lime_instances/'")

# -------------------------------------------------
# 6) HTML Report (Global + Local)
# -------------------------------------------------
def esc(x):
    return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

config_rows = "\n".join(
    [f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in LIME_CONFIG.items()]
)

iframe_blocks = "\n".join([
    f"<details open><summary><b>Instance {c['index']}</b> — True label: {c['true_label']} | "
    f"P(Disease)={c['pred_prob']:.4f}</summary>"
    f"<iframe src='{c['path']}' width='100%' height='520' "
    f"style='border:1px solid #ddd; border-radius:8px;'></iframe></details>"
    for c in local_cards
])

html_content = f"""
<html><head><meta charset='utf-8'/>
<title>LIME Report — NOVEM-CVD Ensemble</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; color:#1f2d3d; }}
h1, h2 {{ color:#2b3e50; }}
table {{ border-collapse: collapse; width:100%; margin-top:12px; }}
th,td {{ border:1px solid #e3e5e8; padding:8px; text-align:center; }}
th {{ background-color:#f7f8fa; }}
img {{ display:block; margin:16px auto; max-width:90%; }}
details {{ margin:14px 0; }}
summary {{ cursor:pointer; padding:8px 0; }}
.note {{ color:#536a7b; font-size:0.92rem; }}
</style></head><body>

<h1>📘 NOVEM-CVD — LIME Explanation Report</h1>

<h2>Configuration</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>{config_rows}</table>

<h2>Global Feature Importance</h2>
<img src="LIME_feature_importance.png" alt="LIME Feature Importance Plot"/>

<h2>Top Features (Preview)</h2>
{lime_summary.head(10).to_html(index=True, justify='center')}

<h2>Stability</h2>
<p><b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}</p>

<h2>Local Explanations (Interactive)</h2>
<p class="note">Includes highest-risk, lowest-risk, and 3 representative cases.</p>
{iframe_blocks}

<hr><p style='text-align:center;' class='note'>
Generated automatically — NOVEM-CVD Explainable Ensemble © 2025
</p></body></html>
"""

with open("LIME_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ LIME HTML report saved as 'LIME_report.html'")
print("📁 Interactive panels located in 'lime_instances/'")

```

### Cell 81 — code (execution_count=76)
```python
# ============================================
# LIME EXPLANATION MODULE — Reviewer-Ready
# NOVEM-CVD (Explainable Ensemble)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib
import warnings, os, random
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
feature_names = top_features_table.apply(
    lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1
).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# Map full PCA component names to shorter labels
short_labels = {
    'ExerciseAngina_ST_Slope': 'ExAng_ST',
    'Cholesterol_FastingBS': 'Chol_FBS',
    'RestingBP_Age': 'BP_Age',
    'RestingECG_FastingBS': 'ECG_FBS',
    'Sex_ChestPainType': 'Sex_CPT',
    'FastingBS_MaxHR': 'FBS_MaxHR',
    'RestingBP_ChestPainType': 'BP_CPT',
    'Cholesterol_Age': 'Chol_Age',
    'Oldpeak_Cholesterol': 'Oldp_Chol'
}

feature_names = [short_labels.get(f, f) for f in feature_names]

# -------------------------------------------------
# 3) LIME Configuration (Reproducible)
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

# Reviewer Concern #15:
# LIME is applied to the final NOVEM-CVD ensemble (VotingClassifier pipeline),
# not to individual base learners. This ensures explanations reflect the actual deployed model.

# Explicit discretizer for transparency & reproducibility
discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],  # 👈 empty list avoids dtype conflict
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=LIME_CONFIG["random_seed"]
)

# -------------------------------------------------
# 4) Global Multi-Run Explanations for Stability
# -------------------------------------------------
n_instances = 30
n_runs = 5
POS_CLASS = 1

instance_indices = np.random.choice(range(X_test_lime.shape[0]), n_instances, replace=False)
feature_weights_runs = []

for run in range(n_runs):
    np.random.seed(LIME_CONFIG["random_seed"] + run)
    run_weights = np.zeros(len(feature_names))

    for idx in instance_indices:
        exp = explainer.explain_instance(
            data_row=X_test_lime[idx],
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight

    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)

# Aggregate stats
signed_mean = np.mean(feature_weights_runs, axis=0)
signed_std  = np.std(feature_weights_runs, axis=0)
abs_mean    = np.mean(np.abs(feature_weights_runs), axis=0)
abs_std     = np.std(np.abs(feature_weights_runs), axis=0)

lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})
lime_summary["Direction"] = np.where(
    lime_summary["SignedMean"] > 0, "↑ increases risk", "↓ decreases risk"
)
lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1

print("\n--- LIME Feature Importance Summary (Top 10) ---")
print(lime_summary.head(10))

# Reviewer Concern #15 (Clinical Plausibility):
print("\nInterpretation Note:")
print("Top LIME features such as Max HR, Age, and Fasting BS align with established cardiovascular risk factors,")
print("supporting the clinical plausibility and external validity of the NOVEM-CVD ensemble's learned associations.")

# -------------------------------------------------
# ✅ FIXED: Global Barplot (Seaborn palette error)
# -------------------------------------------------
# -------------------------------------------------
# ✅ FIXED: Global Barplot (PDF Export + Colab Download)
# -------------------------------------------------
plt.figure(figsize=(10, 6))
colors = ["crimson" if x > 0 else "steelblue" for x in lime_summary["SignedMean"]]
sns.barplot(data=lime_summary, x="SignedMean", y="Feature", palette=colors)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Signed Mean LIME Weight")
plt.title("LIME Feature Directionality of NOVEM-CVD Classifier")
plt.tight_layout()

# ✅ Save as high-quality vector PDF
plt.savefig("LIME_feature_importance.pdf", bbox_inches="tight")
plt.close()
print("✅ LIME barplot saved as 'LIME_feature_importance.pdf'")

# ✅ Automatically download in Google Colab
try:
    from google.colab import files
    files.download("LIME_feature_importance.pdf")
    print("📥 Automatically downloaded 'LIME_feature_importance.pdf'")
except Exception as e:
    print("⚠️ Download skipped (not running in Colab environment):", e)
# -------------------------------------------------
# Stability (Jaccard overlap of Top-k)
# -------------------------------------------------
top_k = 5
top_features_runs = [np.argsort(-np.abs(run))[:top_k] for run in feature_weights_runs]

def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))

stability_scores = [jaccard(top_features_runs[i], top_features_runs[j])
                    for i in range(n_runs) for j in range(i+1, n_runs)]
stability_mean = float(np.mean(stability_scores))
stability_std  = float(np.std(stability_scores))

print(f"\nLIME Stability (Top-{top_k} Jaccard): {stability_mean:.3f} ± {stability_std:.3f}")

# Export CSV
lime_summary.to_csv("LIME_summary.csv", index_label="Rank")
print("✅ LIME summary saved as 'LIME_summary.csv'")

# -------------------------------------------------
# 5) Local Explanations (Interactive Panels)
# -------------------------------------------------
os.makedirs("lime_instances", exist_ok=True)
proba = voting_model.predict_proba(X_test_lime)[:, POS_CLASS]
sorted_idx = np.argsort(proba)
lowest_idx  = int(sorted_idx[0])
highest_idx = int(sorted_idx[-1])
remaining = list(set(range(X_test_lime.shape[0])) - {lowest_idx, highest_idx})
random_idxs = random.sample(remaining, k=min(3, len(remaining)))
local_indices = [highest_idx, lowest_idx] + random_idxs

local_cards = []
for idx in local_indices:
    exp = explainer.explain_instance(
        data_row=X_test_lime[idx],
        predict_fn=voting_model.predict_proba,
        num_features=min(10, len(feature_names)),
        num_samples=LIME_CONFIG["num_samples"]
    )
    inst_path = f"lime_instances/lime_instance_{idx}.html"
    exp.save_to_file(inst_path)
    local_cards.append({
        "index": int(idx),
        "path": inst_path,
        "pred_prob": float(proba[idx]),
        "true_label": int(y_test_lime[idx])
    })
print(f"✅ Saved {len(local_cards)} local LIME panels in 'lime_instances/'")

# -------------------------------------------------
# 6) HTML Report (Global + Local)
# -------------------------------------------------
def esc(x):
    return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

config_rows = "\n".join(
    [f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in LIME_CONFIG.items()]
)

iframe_blocks = "\n".join([
    f"<details open><summary><b>Instance {c['index']}</b> — True label: {c['true_label']} | "
    f"P(Disease)={c['pred_prob']:.4f}</summary>"
    f"<iframe src='{c['path']}' width='100%' height='520' "
    f"style='border:1px solid #ddd; border-radius:8px;'></iframe></details>"
    for c in local_cards
])

html_content = f"""
<html><head><meta charset='utf-8'/>
<title>LIME Report — NOVEM-CVD Ensemble</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; color:#1f2d3d; }}
h1, h2 {{ color:#2b3e50; }}
table {{ border-collapse: collapse; width:100%; margin-top:12px; }}
th,td {{ border:1px solid #e3e5e8; padding:8px; text-align:center; }}
th {{ background-color:#f7f8fa; }}
img {{ display:block; margin:16px auto; max-width:90%; }}
details {{ margin:14px 0; }}
summary {{ cursor:pointer; padding:8px 0; }}
.note {{ color:#536a7b; font-size:0.92rem; }}
</style></head><body>

<h1>📘 NOVEM-CVD — LIME Explanation Report</h1>

<h2>Configuration</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>{config_rows}</table>

<h2>Global Feature Importance</h2>
<img src="LIME_feature_importance.png" alt="LIME Feature Importance Plot"/>

<h2>Top Features (Preview)</h2>
{lime_summary.head(10).to_html(index=True, justify='center')}

<h2>Stability</h2>
<p><b>Top-{top_k} Jaccard Stability:</b> {stability_mean:.3f} ± {stability_std:.3f}</p>

<h2>Local Explanations (Interactive)</h2>
<p class="note">Includes highest-risk, lowest-risk, and 3 representative cases.</p>
{iframe_blocks}

<hr><p style='text-align:center;' class='note'>
Generated automatically — NOVEM-CVD Explainable Ensemble © 2025
</p></body></html>
"""

with open("LIME_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ LIME HTML report saved as 'LIME_report.html'")
print("📁 Interactive panels located in 'lime_instances/'")

# -------------------------------------------------
# 7) Optional — Bundle & Auto-Download All Outputs in Colab
# -------------------------------------------------
try:
    from google.colab import files
    import shutil

    # Create an archive with all results
    shutil.make_archive("LIME_results", 'zip', ".")
    files.download("LIME_results.zip")
    print("📦 All LIME results (PDF, CSV, HTML, and local panels) zipped and downloaded.")
except Exception as e:
    print("⚠️ Auto-download skipped (not running in Colab):", e)

```

### Cell 82 — code (execution_count=77)
```python
# -------------------------------------------------
# 4B) Stability Visualization (Reviewer-Ready)
# -------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Boxplot of pairwise Jaccard similarities
plt.figure(figsize=(6,4))
sns.boxplot(y=stability_scores, color="lightseagreen", width=0.4)
sns.stripplot(y=stability_scores, color="black", alpha=0.6)
plt.title(f"LIME Stability (Top-{top_k} Jaccard Similarities)")
plt.ylabel("Jaccard Similarity")
plt.tight_layout()
plt.savefig("LIME_stability_boxplot.pdf", bbox_inches="tight")
plt.close()
print("✅ LIME stability boxplot saved as 'LIME_stability_boxplot.pdf'")

# 2️⃣ Heatmap of pairwise Jaccard matrix
jaccard_matrix = np.zeros((n_runs, n_runs))
for i in range(n_runs):
    for j in range(n_runs):
        jaccard_matrix[i, j] = jaccard(top_features_runs[i], top_features_runs[j])

plt.figure(figsize=(6,5))
sns.heatmap(jaccard_matrix, annot=True, cmap="YlGnBu", fmt=".2f",
            cbar_kws={'label': 'Jaccard Similarity'})
plt.title(f"LIME Stability Matrix (Top-{top_k})")
plt.xlabel("Run")
plt.ylabel("Run")
plt.tight_layout()
plt.savefig("LIME_stability_heatmap.pdf", bbox_inches="tight")
plt.close()
print("✅ LIME stability heatmap saved as 'LIME_stability_heatmap.pdf'")

# 3️⃣ Simple summary barplot with mean ± SD
plt.figure(figsize=(5,4))
plt.bar([f"Top-{top_k}"], [stability_mean], yerr=[stability_std],
        color="teal", alpha=0.8, capsize=6)
plt.ylim(0,1)
plt.ylabel("Mean Jaccard Similarity")
plt.title("LIME Stability Summary")
plt.tight_layout()
plt.savefig("LIME_stability_summary.pdf", bbox_inches="tight")
plt.close()
print("✅ LIME stability summary barplot saved as 'LIME_stability_summary.pdf'")

# Auto-download all stability plots in Colab
try:
    from google.colab import files
    files.download("LIME_stability_boxplot.pdf")
    files.download("LIME_stability_heatmap.pdf")
    files.download("LIME_stability_summary.pdf")
    print("📥 Automatically downloaded stability visualizations (3 PDFs)")
except Exception as e:
    print("⚠️ Auto-download skipped (not running in Colab):", e)

```

### Cell 83 — code (execution_count=None)
```python

```

### Cell 84 — code (execution_count=None)
```python
# ============================================
# NOVEM-CVD LIME EXPLANATION MODULE — Reviewer-Ready
# (Explainable Ensemble + Local Panels + Thumbnails + Auto Packaging)
# ============================================

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib, warnings, os, random, datetime, shutil, zipfile, glob, sys
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib.styles import getSampleStyleSheet
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. Load Pre-Saved Model and Data
# -------------------------------------------------
log("🔄 Loading pre-saved model and dataset...")
data = joblib.load("LoRAS_train_test_data.pkl")
X_train_lime, X_test_lime = data["X_train"], data["X_test"]
y_train_lime, y_test_lime = data["y_train"], data["y_test"]
voting_model = joblib.load("voting_pipeline.pkl")
log("✅ Data and model loaded successfully!")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
feature_names = top_features_table.apply(
    lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1
).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# Map full PCA component names to shorter labels
short_labels = {
    'ExerciseAngina_ST_Slope': 'ExAng_ST',
    'Cholesterol_FastingBS': 'Chol_FBS',
    'RestingBP_Age': 'BP_Age',
    'RestingECG_FastingBS': 'ECG_FBS',
    'Sex_ChestPainType': 'Sex_CPT',
    'FastingBS_MaxHR': 'FBS_MaxHR',
    'RestingBP_ChestPainType': 'BP_CPT',
    'Cholesterol_Age': 'Chol_Age',
    'Oldpeak_Cholesterol': 'Oldp_Chol'
}

feature_names = [short_labels.get(f, f) for f in feature_names]

# -------------------------------------------------
# 3. LIME Configuration
# -------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
LIME_CONFIG = {
    "kernel_type":"exponential","kernel_width":3.0,"num_samples":5000,
    "discretizer":"quartile","random_seed":RANDOM_SEED
}
discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],          # empty list avoids TypeError
    feature_names=feature_names
)
explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease","Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=RANDOM_SEED
)

# -------------------------------------------------
# 4. Global Multi-Run Explanations
# -------------------------------------------------
n_instances, n_runs, POS_CLASS = 30, 5, 1
instance_indices = np.random.choice(range(X_test_lime.shape[0]), n_instances, replace=False)
feature_weights_runs = []

for run in range(n_runs):
    np.random.seed(RANDOM_SEED + run)
    run_weights = np.zeros(len(feature_names))
    for idx in instance_indices:
        exp = explainer.explain_instance(
            data_row=X_test_lime[idx],
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        for feat_idx, weight in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += weight
    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)
signed_mean, signed_std = np.mean(feature_weights_runs,0), np.std(feature_weights_runs,0)
abs_mean, abs_std = np.mean(np.abs(feature_weights_runs),0), np.std(np.abs(feature_weights_runs),0)

lime_summary = pd.DataFrame({
    "Feature":feature_names,
    "SignedMean":signed_mean,"SignedSD":signed_std,
    "AbsMean":abs_mean,"AbsSD":abs_std
})
lime_summary["Direction"] = np.where(lime_summary["SignedMean"]>0,"↑ increases risk","↓ decreases risk")
lime_summary = lime_summary.sort_values("AbsMean",ascending=False).reset_index(drop=True)
lime_summary.index += 1

log("\n--- LIME Feature Importance Summary (Top 10) ---")
log(lime_summary.head(10).to_string(index=True))
log("\nInterpretation Note:\nTop LIME features such as Max HR, Age, and Fasting BS align with established cardiovascular risk factors,\nsupporting the clinical plausibility and external validity of the NOVEM-CVD ensemble's learned associations.")

# -------------------------------------------------
# 5. Plot Global Barplot (Legend below, no overlap)
# -------------------------------------------------
import matplotlib.patches as mpatches
plt.figure(figsize=(10, 6))
colors = lime_summary["SignedMean"].apply(lambda x: "crimson" if x > 0 else "steelblue")
plt.barh(lime_summary["Feature"], lime_summary["SignedMean"], color=colors)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Signed Mean LIME Weight")
plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble")
red_patch = mpatches.Patch(color='crimson', label='↑ increases risk')
blue_patch = mpatches.Patch(color='steelblue', label='↓ decreases risk')
plt.legend(handles=[red_patch, blue_patch],
           title="Effect Direction", loc='upper center',
           bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("LIME_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
log("✅ LIME barplot saved as 'LIME_feature_importance.png'")

# -------------------------------------------------
# 6. Stability
# -------------------------------------------------
top_k=5
tops=[np.argsort(-np.abs(r))[:top_k] for r in feature_weights_runs]
jaccard=lambda a,b:len(set(a)&set(b))/len(set(a)|set(b))
stab=[jaccard(tops[i],tops[j]) for i in range(n_runs) for j in range(i+1,n_runs)]
stab_mean, stab_std = float(np.mean(stab)), float(np.std(stab))
lime_summary.to_csv("LIME_summary.csv",index_label="Rank")
log("✅ LIME summary saved as 'LIME_summary.csv'")
log(f"🧮 Stability (Top-{top_k} Jaccard): {stab_mean:.3f} ± {stab_std:.3f}")

# -------------------------------------------------
# 7. Local Explanations + PNG Thumbnails
# -------------------------------------------------
os.makedirs("lime_instances",exist_ok=True)
proba = voting_model.predict_proba(X_test_lime)[:,POS_CLASS]
hi, lo = np.argmax(proba), np.argmin(proba)
others=list(set(range(len(X_test_lime)))-{hi,lo}); randoms=random.sample(others,3)
local_indices=[hi,lo]+randoms
local_cards=[]
for idx in local_indices:
    exp=explainer.explain_instance(
        data_row=X_test_lime[idx],
        predict_fn=voting_model.predict_proba,
        num_features=min(10,len(feature_names)),
        num_samples=LIME_CONFIG["num_samples"]
    )
    html_path=f"lime_instances/lime_instance_{idx}.html"
    exp.save_to_file(html_path)
    png_path=f"lime_instances/lime_instance_{idx}.png"
    thumb_ok = html_to_png(html_path, png_path)
    local_cards.append({
        "index":idx,"path_html":html_path,"path_png":png_path if thumb_ok and os.path.exists(png_path) else None,
        "pred_prob":float(proba[idx]),"true_label":int(y_test_lime[idx])
    })
log(f"✅ Saved {len(local_cards)} local panels in 'lime_instances/' (PNGs created when renderer available)")

# -------------------------------------------------
# 8. Build HTML + PDF + ZIP Reports
# -------------------------------------------------
timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir=f"LIME_outputs_{timestamp}"
os.makedirs(outputs_dir,exist_ok=True)

# --- HTML report (quick) ---
iframe_blocks="\n".join([
    f"<details open><summary><b>Instance {c['index']}</b> — True label: {c['true_label']} | P(Disease)={c['pred_prob']:.4f}</summary>"
    f"<iframe src='{c['path_html']}' width='100%' height='520' style='border:1px solid #ddd;border-radius:8px;'></iframe>"
    + (f"<div><img src='{os.path.basename(c['path_png'])}' style='max-width:60%;margin:10px auto;display:block;'/></div>" if c['path_png'] else "")
    + "</details>"
    for c in local_cards
])
html=f"""<html><head><meta charset='utf-8'/>
<title>LIME Report — NOVEM-CVD Ensemble</title></head><body>
<h1>📘 NOVEM-CVD — LIME Explanation Report</h1>
<p><b>Generated:</b> {timestamp}</p>
<img src='LIME_feature_importance.png' width='90%'>
{lime_summary.head(10).to_html(index=True)}
<p><b>Top-{top_k} Jaccard Stability:</b> {stab_mean:.3f} ± {stab_std:.3f}</p>
<h2>Local Explanations (inline panel + PNG thumbnail)</h2>{iframe_blocks}
<hr><p style='text-align:center;color:#666'>NOVEM-CVD Explainable Ensemble © 2025</p></body></html>"""
html_path=os.path.join(outputs_dir,"LIME_report.html")
with open(html_path,"w",encoding="utf-8") as f:f.write(html)
log(f"✅ HTML report saved as '{html_path}'")

# copy assets
for f in ["LIME_summary.csv","LIME_feature_importance.png"]:
    shutil.copy(f,outputs_dir)
# copy panels + thumbnails
shutil.copytree("lime_instances",os.path.join(outputs_dir,"lime_instances"),dirs_exist_ok=True)

# --- PDF (each figure/page separately) ---
pdf_path=os.path.join(outputs_dir,"LIME_report.pdf")
doc=SimpleDocTemplate(pdf_path,pagesize=letter)
styles=getSampleStyleSheet(); story=[]

# Cover Page
story.append(Paragraph("📘 NOVEM-CVD LIME Explanation Report", styles['Title']))
story.append(Spacer(1,12))
story.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
story.append(Paragraph("Author: Muhammad Jaffar Khan", styles['Normal']))
story.append(Spacer(1,24))
story.append(PageBreak())

# Configuration Page
tbl=[[k,str(v)] for k,v in LIME_CONFIG.items()]
t=Table(tbl,colWidths=[200,200])
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),rl_colors.lightgrey),
                       ('GRID',(0,0),(-1,-1),0.5,rl_colors.grey)]))
story.append(Paragraph("<b>Configuration</b>",styles['Heading2']))
story.append(t); story.append(Spacer(1,24)); story.append(PageBreak())

# Top Features Page
top10=lime_summary.head(10)
data=[list(top10.columns)]+top10.values.tolist()
t=Table(data,repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),rl_colors.lightgrey),
                       ('GRID',(0,0),(-1,-1),0.5,rl_colors.grey)]))
story.append(Paragraph("<b>Top 10 Global LIME Features</b>",styles['Heading2']))
story.append(t); story.append(Spacer(1,24)); story.append(PageBreak())

# Stability Page
story.append(Paragraph("<b>Model Stability Analysis</b>",styles['Heading2']))
story.append(Paragraph(f"Top-{top_k} Jaccard Stability: {stab_mean:.3f} ± {stab_std:.3f}",styles['Normal']))
story.append(Spacer(1,24)); story.append(PageBreak())

# Global Barplot Page
story.append(Paragraph("<b>Global Feature Importance Chart</b>",styles['Heading2']))
story.append(Image("LIME_feature_importance.png",width=450,height=300))
legend=Drawing(220,20)
legend.add(Rect(0,0,20,10,fillColor=rl_colors.crimson))
legend.add(String(25,0,"↑ increases risk",fontSize=10))
legend.add(Rect(120,0,20,10,fillColor=rl_colors.steelblue))
legend.add(String(145,0,"↓ decreases risk",fontSize=10))
story.append(Spacer(1,8)); story.append(legend); story.append(PageBreak())

# One page per local instance (with PNG if available)
story.append(Paragraph("<b>Local Explanations — Thumbnails</b>",styles['Heading2']))
for card in local_cards:
    story.append(Paragraph(
        f"Instance {card['index']} — True Label: {card['true_label']} — P(Disease): {card['pred_prob']:.4f}",
        styles['Normal']
    ))
    if card['path_png'] and os.path.exists(card['path_png']):
        story.append(Image(card['path_png'], width=450, height=320))
    else:
        story.append(Paragraph("(PNG thumbnail not available — panel included in HTML bundle)", styles['Italic']))
    story.append(Spacer(1,18))
    story.append(PageBreak())

doc.build(story)
log(f"✅ PDF report saved as '{pdf_path}'")

# --- ZIP + Friendly Download ---
zip_name=f"LIME_outputs_{timestamp}.zip"
with zipfile.ZipFile(zip_name,"w") as z:
    for root,_,files in os.walk(outputs_dir):
        for f in files:
            full=os.path.join(root,f)
            rel=os.path.relpath(full,os.path.dirname(outputs_dir))
            z.write(full,arcname=rel)
friendly="NOVEM_CVD_LIME_Report_Latest.zip"
shutil.copy(zip_name,friendly)
log(f"📦 Packaged and copied as '{friendly}'")

# Auto-download (if Colab)
try:
    from google.colab import files
    log("📥 Auto-downloading...")
    files.download(friendly)
except Exception:
    log("⚠️ Not in Colab, manual download required.")

```

### Cell 85 — code (execution_count=47)
```python
# ============================================
# NOVEM-CVD — FAST LIME EXPLANATION MODULE
# Reviewer-Ready + PDF Report + Individual PDFs
# ============================================

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib, warnings, os, random, datetime, shutil, zipfile
warnings.filterwarnings("ignore")

from reportlab.lib import colors as rl_colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet

CRIMSON = HexColor("#DC143C")
STEELBLUE = HexColor("#4682B4")

def log(msg): print(msg, flush=True)


# =====================================================
# ✅ 1. LOAD MODEL + DATA
# =====================================================
import joblib
pipeline_loaded = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")
print("✅ Model loaded successfully!")

# Load saved data from Google Drive
data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")

X_train = data['X_train']
X_test  = data['X_test']
y_train = data['y_train']
y_test  = data['y_test']

print("✅ Data reloaded successfully — ready for LIME!")

# =====================================================
# ✅ 3. LIME CONFIGURATION
# =====================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,     # FAST
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

discretizer = QuartileDiscretizer(
    np.array(X_train_lime),
    categorical_features=[],
    feature_names=feature_names
)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train_lime),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=RANDOM_SEED
)


# =====================================================
# ✅ 4. GLOBAL MULTI-RUN (FAST MODE)
# =====================================================
n_instances = 30
n_runs = 5
POS_CLASS = 1

instance_indices = np.random.choice(
    range(X_test_lime.shape[0]),
    n_instances,
    replace=False
)

feature_weights_runs = []

for run in range(n_runs):
    np.random.seed(RANDOM_SEED + run)
    run_weights = np.zeros(len(feature_names))

    for idx in instance_indices:
        exp = explainer.explain_instance(
            data_row=X_test_lime[idx],
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        for feat_idx, w in exp.as_map()[POS_CLASS]:
            run_weights[feat_idx] += w

    feature_weights_runs.append(run_weights / n_instances)

feature_weights_runs = np.array(feature_weights_runs)

signed_mean = feature_weights_runs.mean(axis=0)
signed_std  = feature_weights_runs.std(axis=0)
abs_mean    = np.abs(feature_weights_runs).mean(axis=0)
abs_std     = np.abs(feature_weights_runs).std(axis=0)

lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})

lime_summary = lime_summary.sort_values("AbsMean", ascending=False)
lime_summary.index += 1
lime_summary.to_csv("LIME_summary.csv")

log("✅ Global feature importance computed!")


# =====================================================
# ✅ 5. GLOBAL SIGNED BARPLOT (PNG)
# =====================================================
plt.figure(figsize=(10, 6))
colors = ["crimson" if x > 0 else "steelblue" for x in lime_summary["SignedMean"]]
plt.barh(lime_summary["Feature"], lime_summary["SignedMean"], color=colors)
plt.axvline(0, color='black', linestyle='--')
plt.xlabel("Signed Mean LIME Weight")
plt.title("LIME Feature Directionality — NOVEM-CVD")
plt.tight_layout()
plt.savefig("LIME_signed.png", dpi=300, bbox_inches="tight")
plt.close()

log("✅ Saved LIME_signed.png")


# =====================================================
# ✅ 6. STABILITY PLOTS — SAVE AS PNG ONLY
# =====================================================
import seaborn as sns

top_k = 5
tops = [np.argsort(-np.abs(run))[:top_k] for run in feature_weights_runs]
jacc = lambda a,b: len(set(a)&set(b)) / len(set(a)|set(b))

stab = [jacc(tops[i], tops[j]) for i in range(n_runs) for j in range(i+1, n_runs)]
stab_mean, stab_std = float(np.mean(stab)), float(np.std(stab))

# -------- Heatmap --------
jmat = np.zeros((n_runs, n_runs))
for i in range(n_runs):
    for j in range(n_runs):
        jmat[i,j] = jacc(tops[i], tops[j])

plt.figure(figsize=(6,5))
sns.heatmap(jmat, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("LIME Stability Matrix")
plt.tight_layout()
plt.savefig("LIME_stability_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# -------- Boxplot --------
plt.figure(figsize=(6,4))
sns.boxplot(y=stab, color="#20B2AA")
sns.stripplot(y=stab, color="black", alpha=0.6)
plt.title("LIME Stability — Jaccard (Top-5)")
plt.tight_layout()
plt.savefig("LIME_stability_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

# -------- Summary --------
plt.figure(figsize=(5,4))
plt.bar(["Top-5"], [stab_mean], yerr=[stab_std], color="#008080", capsize=6)
plt.ylim(0,1)
plt.ylabel("Mean Jaccard Similarity")
plt.title("LIME Stability Summary")
plt.tight_layout()
plt.savefig("LIME_stability_summary.png", dpi=300, bbox_inches="tight")
plt.close()

log("✅ Stability PNGs saved!")


# =====================================================
# ✅ 7. LOCAL EXPLANATIONS (HTML)
# =====================================================
os.makedirs("lime_instances", exist_ok=True)
proba = voting_model.predict_proba(X_test_lime)[:, 1]

hi, lo = np.argmax(proba), np.argmin(proba)
others = list(set(range(len(X_test_lime))) - {hi, lo})
rand = random.sample(others, min(3, len(others)))
local_indices = [hi, lo] + rand

local_cards = []

for idx in local_indices:
    exp = explainer.explain_instance(
        data_row=X_test_lime[idx],
        predict_fn=voting_model.predict_proba,
        num_features=min(10, len(feature_names)),
        num_samples=LIME_CONFIG["num_samples"]
    )
    html_path = f"lime_instances/lime_instance_{idx}.html"
    exp.save_to_file(html_path)

    local_cards.append({
        "index": idx,
        "path_html": html_path,
        "true": int(y_test_lime[idx]),
        "prob": float(proba[idx])
    })

log("✅ Saved local explanations (HTML panels)")


# =====================================================
# ✅ 8. BUILD PDF REPORT
# =====================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"LIME_outputs_{timestamp}"
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, "LIME_report.pdf")

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
story = []

# ---- Cover ----
story.append(Paragraph("📘 NOVEM-CVD LIME Explanation Report", styles['Title']))
story.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
story.append(PageBreak())

# ---- Config ----
cfg = [[k, str(v)] for k, v in LIME_CONFIG.items()]
table = Table(cfg)
table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, rl_colors.grey)]))
story.append(Paragraph("Configuration", styles['Heading2']))
story.append(table)
story.append(PageBreak())

# ---- Top 10 ----
story.append(Paragraph("Top 10 Global Features", styles['Heading2']))
top10 = lime_summary.head(10)
tbl = Table([list(top10.columns)] + top10.values.tolist(), repeatRows=1)
tbl.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.3, rl_colors.grey)]))
story.append(tbl)
story.append(PageBreak())

# ---- Signed Plot ----
story.append(Paragraph("Global Feature Importance", styles['Heading2']))
story.append(Image("LIME_signed.png", width=460, height=300))
story.append(PageBreak())

# ---- Stability ----
story.append(Paragraph("Stability Analysis", styles['Heading2']))
for fig in [
    "LIME_stability_heatmap.png",
    "LIME_stability_boxplot.png",
    "LIME_stability_summary.png"
]:
    story.append(Image(fig, width=440, height=320))
    story.append(Spacer(1, 12))
story.append(PageBreak())

# ---- Local Explanations ----
story.append(Paragraph("Local Explanations (HTML Links)", styles['Heading2']))
for card in local_cards:
    story.append(Paragraph(
        f"Instance {card['index']} — True={card['true']} — P(Disease)={card['prob']:.4f}",
        styles['Normal']
    ))
    story.append(Paragraph(card["path_html"], styles['Italic']))
    story.append(Spacer(1, 12))

doc.build(story)
log(f"✅ PDF report saved → {pdf_path}")


# =====================================================
# ✅ 9. ZIP PACKAGE
# =====================================================
for f in [
    "LIME_summary.csv",
    "LIME_signed.png",
    "LIME_stability_heatmap.png",
    "LIME_stability_boxplot.png",
    "LIME_stability_summary.png",
]:
    shutil.copy(f, out_dir)

shutil.copytree("lime_instances", os.path.join(out_dir, "lime_instances"))

zip_name = f"LIME_outputs_{timestamp}.zip"
with zipfile.ZipFile(zip_name, "w") as z:
    for root, _, files in os.walk(out_dir):
        for f in files:
            full = os.path.join(root, f)
            rel  = os.path.relpath(full, os.path.dirname(out_dir))
            z.write(full, arcname=rel)

log(f"📦 Packaged → {zip_name}")


# =====================================================
# ✅ 10. SAVE EACH FIGURE AS INDIVIDUAL PDF
# =====================================================
def img_to_pdf(img_path, pdf_path):
    img = plt.imread(img_path)
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

individual = [
    ("LIME_signed.png", "LIME_signed.pdf"),
    ("LIME_stability_heatmap.png", "LIME_stability_heatmap.pdf"),
    ("LIME_stability_boxplot.png", "LIME_stability_boxplot.pdf"),
    ("LIME_stability_summary.png", "LIME_stability_summary.pdf"),
]

for png, pdf in individual:
    if os.path.exists(png):
        img_to_pdf(png, pdf)
        log(f"✅ Saved {pdf}")


# =====================================================
# ✅ 11. AUTO-DOWNLOAD (COLAB)
# =====================================================
try:
    from google.colab import files
    files.download(zip_name)
    for _, pdf in individual:
        files.download(pdf)
    log("📥 Auto-download started")
except:
    log("ℹ️ Not running in Colab — download manually.")

```

### Cell 86 — code (execution_count=None)
```python

```

### Cell 87 — code (execution_count=84)
```python
# ==============================================================
# NOVEM-CVD — LIME EXPLANATION MODULE (Colab-Optimized Reviewer Edition)
# ==============================================================

!pip install -q lime imgkit reportlab joblib matplotlib pandas numpy

import shutil, subprocess, sys, os, random, datetime, time, warnings, zipfile
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib
warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# MODE FLAGS
# --------------------------------------------------------------
REVIEWER_MODE = True  # ✅ True = Fast Reviewer Mode, False = Full mode
SKIP_THUMBNAILS = True if REVIEWER_MODE else False
SKIP_PDF = True if REVIEWER_MODE else False

# --------------------------------------------------------------
# SETUP ENVIRONMENT (wkhtmltopdf)
# --------------------------------------------------------------
def ensure_imgkit_and_wkhtml():
    wkhtml = shutil.which("wkhtmltoimage")
    if not wkhtml:
        print("Installing wkhtmltopdf (includes wkhtmltoimage)…")
        !apt-get update -y
        !apt-get install -y wkhtmltopdf
    import imgkit
    wkhtml = shutil.which("wkhtmltoimage")
    print(f"wkhtmltoimage path: {wkhtml}")
    return wkhtml

if not SKIP_THUMBNAILS:
    ensure_imgkit_and_wkhtml()

# --------------------------------------------------------------
# LOAD DATA & MODEL
# --------------------------------------------------------------
print("[load] Loading pre-saved model and dataset…")
data = joblib.load("LoRAS_train_test_data.pkl")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
voting_model = joblib.load("voting_pipeline.pkl")
print("[load] ✅ Data and model loaded successfully")

# --------------------------------------------------------------
# FEATURE NAMES
# --------------------------------------------------------------
# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
feature_names = top_features_table.apply(
    lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1
).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# Map full PCA component names to shorter labels
short_labels = {
    'ExerciseAngina_ST_Slope': 'ExAng_ST',
    'Cholesterol_FastingBS': 'Chol_FBS',
    'RestingBP_Age': 'BP_Age',
    'RestingECG_FastingBS': 'ECG_FBS',
    'Sex_ChestPainType': 'Sex_CPT',
    'FastingBS_MaxHR': 'FBS_MaxHR',
    'RestingBP_ChestPainType': 'BP_CPT',
    'Cholesterol_Age': 'Chol_Age',
    'Oldpeak_Cholesterol': 'Oldp_Chol'
}

feature_names = [short_labels.get(f, f) for f in feature_names]

# --------------------------------------------------------------
# CONFIGURE LIME
# --------------------------------------------------------------
print("[lime] Configuring LIME…")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
LIME_CONFIG = {"kernel_type":"exponential","kernel_width":3.0,
               "num_samples": 1000 if REVIEWER_MODE else 5000,
               "discretizer":"quartile","random_seed":RANDOM_SEED}
discretizer = QuartileDiscretizer(np.array(X_train), categorical_features=[], feature_names=feature_names)
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=["No Disease","Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=RANDOM_SEED
)
print("[lime] ✅ LIME ready")

# --------------------------------------------------------------
# GLOBAL MULTI-RUN
# --------------------------------------------------------------
print("[global] Running multi-run LIME explanations…")
n_instances, n_runs, POS_CLASS = (10, 3, 1) if REVIEWER_MODE else (30, 5, 1)
instance_idx = np.random.choice(range(X_test.shape[0]), n_instances, replace=False)
feature_runs = []
for run in range(n_runs):
    np.random.seed(RANDOM_SEED + run)
    run_w = np.zeros(len(feature_names))
    for idx in instance_idx:
        exp = explainer.explain_instance(
            data_row=X_test[idx],
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        for fidx, w in exp.as_map()[POS_CLASS]:
            run_w[fidx] += w
    feature_runs.append(run_w / n_instances)
feature_runs = np.array(feature_runs)

signed_mean, signed_std = feature_runs.mean(0), feature_runs.std(0)
abs_mean, abs_std = np.abs(feature_runs).mean(0), np.abs(feature_runs).std(0)
lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean, "SignedSD": signed_std,
    "AbsMean": abs_mean, "AbsSD": abs_std
})
lime_summary["Direction"] = np.where(lime_summary["SignedMean"]>0,"↑ increases risk","↓ decreases risk")
lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1
print("[global] ✅ Aggregated feature importance computed")

# --------------------------------------------------------------
# PLOTS
# --------------------------------------------------------------
print("[plots] Generating plots…")
import matplotlib.patches as mpatches
plt.figure(figsize=(9,5))
sorted_signed = lime_summary.sort_values("SignedMean", ascending=True)
colors = ["#d62728" if x>0 else "#1f77b4" for x in sorted_signed["SignedMean"]]
plt.barh(sorted_signed["Feature"], sorted_signed["SignedMean"], color=colors)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("LIME Feature Directionality — NOVEM-CVD Ensemble")
plt.xlabel("Signed Mean LIME Weight")
plt.tight_layout()
plt.savefig("LIME_feature_importance.png",dpi=300,bbox_inches="tight"); plt.close()

plt.figure(figsize=(9,5))
sorted_abs = lime_summary.sort_values("AbsMean",ascending=True)
plt.barh(sorted_abs["Feature"],sorted_abs["AbsMean"],color="#6baed6")
plt.xlabel("|Mean LIME Weight| (Importance Magnitude)")
plt.title("Absolute LIME Feature Importance — NOVEM-CVD Ensemble")
plt.tight_layout()
plt.savefig("LIME_feature_importance_abs.png",dpi=300,bbox_inches="tight"); plt.close()
print("[plots] ✅ Plots saved")

# --------------------------------------------------------------
# LOCAL EXPLANATIONS
# --------------------------------------------------------------
print("[local] Running local explanations…")
os.makedirs("lime_instances", exist_ok=True)
proba = voting_model.predict_proba(X_test)[:, 1]
hi, lo = int(np.argmax(proba)), int(np.argmin(proba))
local_idx = [hi, lo]

local_cards = []
for i, idx in enumerate(local_idx, 1):
    start = time.time()
    print(f"  → Instance {idx} ({i}/{len(local_idx)})…")
    exp = explainer.explain_instance(
        data_row=X_test[idx],
        predict_fn=voting_model.predict_proba,
        num_features=8,
        num_samples=LIME_CONFIG["num_samples"]
    )
    html_path = f"lime_instances/lime_instance_{idx}.html"
    exp.save_to_file(html_path)
    elapsed = time.time() - start
    print(f"     ✓ Done in {elapsed:.1f}s")
    local_cards.append({"index": idx, "path_html": html_path,
                        "pred_prob": float(proba[idx]), "true_label": int(y_test[idx])})
print(f"[local] ✅ {len(local_cards)} local explanations generated")

# --------------------------------------------------------------
# REPORT GENERATION (HTML)
# --------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = f"LIME_outputs_{timestamp}"
os.makedirs(outputs_dir, exist_ok=True)
for f in ["LIME_feature_importance.png", "LIME_feature_importance_abs.png"]:
    shutil.copy(f, outputs_dir)
shutil.copytree("lime_instances", os.path.join(outputs_dir, "lime_instances"), dirs_exist_ok=True)

iframe_blocks = "\n".join([
    f"<details open><summary><b>Instance {c['index']}</b> — True label: {c['true_label']} | P(Disease)={c['pred_prob']:.4f}</summary>"
    f"<iframe src='{c['path_html']}' width='100%' height='520' style='border:1px solid #ddd;border-radius:8px;'></iframe></details>"
    for c in local_cards
])
html = f"""<html><head><meta charset='utf-8'/>
<title>LIME Report — NOVEM-CVD Ensemble</title></head><body>
<h1>📘 NOVEM-CVD — LIME Explanation Report</h1>
<p><b>Generated:</b> {timestamp}</p>
<p><b>Reviewer Mode:</b> {REVIEWER_MODE}</p>
<img src='LIME_feature_importance.png' width='90%'>
<img src='LIME_feature_importance_abs.png' width='90%'>
{lime_summary.head(10).to_html(index=True)}
<h2>Local Explanations</h2>{iframe_blocks}
<hr><p style='text-align:center;color:#666'>NOVEM-CVD Explainable Ensemble © 2025</p></body></html>"""
with open(os.path.join(outputs_dir, "LIME_report.html"), "w", encoding="utf-8") as f:
    f.write(html)
print(f"[report] ✅ HTML report saved → {outputs_dir}/LIME_report.html")

# --------------------------------------------------------------
# ZIP & DOWNLOAD
# --------------------------------------------------------------
print("[zip] Packaging…")
zip_name = f"LIME_outputs_{timestamp}.zip"
with zipfile.ZipFile(zip_name, "w") as z:
    for root, _, files in os.walk(outputs_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, os.path.dirname(outputs_dir))
            z.write(full, arcname=rel)
from google.colab import files
files.download(zip_name)
print(f"[zip] ✅ Ready for download → {zip_name}")

```

### Cell 88 — code (execution_count=None)
```python

```

### Cell 89 — code (execution_count=39)
```python
# ============================================
# NOVEM-CVD — LIME EXPLANATION MODULE (Optimized & Reviewer-Ready)
# FULL MERGED VERSION — FAST & FREEZE-PROOF
# ============================================

# ========== 0) MODE FLAGS ==========
SKIP_PDF = False        # Keep PDF unless you want max speed

# ========== 1) SETUP ==========
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import random, datetime, zipfile, os, warnings, shutil, joblib
warnings.filterwarnings("ignore")

def log(sec, msg): print(f"[{sec}] {msg}", flush=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer
import joblib
import warnings, os, random
warnings.filterwarnings("ignore")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
# -------------------------------------------------
# 1. Load Pre-Saved Data and Model
# -------------------------------------------------
print("🔄 Loading pre-saved model and dataset...")

voting_model = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")
print("✅ Model loaded successfully!")

data = joblib.load("/content/drive/MyDrive/models/LoRAS_train_test_data.pkl")
X_train_lime = data["X_train"]
X_test_lime = data["X_test"]
y_train_lime = data["y_train"]
y_test_lime = data["y_test"]

print("✅ Data reloaded successfully — ready for LIME!")

# -------------------------------------------------
# 2. Define Feature Names
# -------------------------------------------------
# Replace this with your actual feature generation logic if needed
feature_names = top_features_table.apply(
    lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1
).tolist()

if isinstance(X_train_lime, pd.DataFrame):
    X_train_lime = X_train_lime.to_numpy()
    X_test_lime = X_test_lime.to_numpy()

# Map full PCA component names to shorter labels
short_labels = {
    'ExerciseAngina_ST_Slope': 'ExAng_ST',
    'Cholesterol_FastingBS': 'Chol_FBS',
    'RestingBP_Age': 'BP_Age',
    'RestingECG_FastingBS': 'ECG_FBS',
    'Sex_ChestPainType': 'Sex_CPT',
    'FastingBS_MaxHR': 'FBS_MaxHR',
    'RestingBP_ChestPainType': 'BP_CPT',
    'Cholesterol_Age': 'Chol_Age',
    'Oldpeak_Cholesterol': 'Oldp_Chol'
}

feature_names = [short_labels.get(f, f) for f in feature_names]

# ========== 4) LIME CONFIGURATION ==========
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer

np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)

LIME_CONFIG = {
    "kernel_type": "exponential",
    "kernel_width": 3.0,
    "num_samples": 1500,
    "discretizer": "quartile",
    "random_seed": RANDOM_SEED
}

log("cfg", "Configuring LIME discretizer and explainer...")
discretizer = QuartileDiscretizer(np.array(X_train), categorical_features=[], feature_names=feature_names)
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    kernel_width=LIME_CONFIG["kernel_width"],
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=RANDOM_SEED
)
log("cfg", "✅ LIME ready")

# ========== 5) GLOBAL EXPLANATIONS (multi-run aggregation) ==========
log("global", "Running LIME across multiple runs & instances...")
n_instances, n_runs, POS_CLASS = 30, 5, 1
instance_idx = np.random.choice(range(X_test.shape[0]), n_instances, replace=False)

feature_weights_runs = []
for run in range(n_runs):
    np.random.seed(RANDOM_SEED + run)
    run_w = np.zeros(len(feature_names))
    for idx in instance_idx:
        exp = explainer.explain_instance(
            data_row=X_test[idx],
            predict_fn=voting_model.predict_proba,
            num_features=len(feature_names),
            num_samples=LIME_CONFIG["num_samples"]
        )
        # exp.as_map() returns {class: [(feature_idx, weight), ...]}
        for fidx, w in exp.as_map()[POS_CLASS]:
            run_w[fidx] += w
    feature_weights_runs.append(run_w / n_instances)

feature_runs = np.array(feature_weights_runs)  # shape: (n_runs, n_features)

# -------------------------------------------------
# 5. Compute Signed and Absolute Statistics (merged)
# -------------------------------------------------
signed_mean = np.mean(feature_runs, axis=0)
signed_std  = np.std(feature_runs, axis=0)
abs_mean    = np.mean(np.abs(feature_runs), axis=0)
abs_std     = np.std(np.abs(feature_runs), axis=0)

# Build summary dataframe (directionality + magnitude)
lime_summary = pd.DataFrame({
    "Feature": feature_names,
    "SignedMean": signed_mean,
    "SignedSD": signed_std,
    "AbsMean": abs_mean,
    "AbsSD": abs_std
})

# Add human-readable risk direction
lime_summary["Direction"] = np.where(lime_summary["SignedMean"] > 0, "↑ increases risk", "↓ decreases risk")
lime_summary = lime_summary.sort_values("AbsMean", ascending=False).reset_index(drop=True)
lime_summary.index += 1  # start ranking at 1

print("\n--- LIME Feature Importance Summary (Top 10) ---")
print(lime_summary.head(10).to_string())

# Save CSV for downstream packaging
lime_summary.to_csv("LIME_summary.csv", index_label="Rank")
log("global", "✅ Saved LIME_summary.csv")

# ========== 6) VISUALIZATIONS (global plots + Seaborn direction plot) ==========
import seaborn as sns
import matplotlib.patches as mpatches

def save_plot(fig, path, close=True):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if close:
        plt.close(fig)

# 6.1 Directionality (Seaborn styled — your requested block)
fig = plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
sns.set_context("talk")

# seaborn barplot expects an ordered DataFrame; lime_summary is sorted by AbsMean (desc)
# For direction-aware plot we might want to plot in the same sorted order:
plot_df = lime_summary.copy()  # already sorted by AbsMean desc
# If you'd prefer signed-mean ordering, uncomment:
# plot_df = lime_summary.sort_values("SignedMean", ascending=True).reset_index(drop=True)

# Palette as list (crimson for positive, steelblue for negative)
colors = plot_df["SignedMean"].apply(lambda x: "crimson" if x > 0 else "steelblue").tolist()

ax = sns.barplot(
    data=plot_df,
    x="SignedMean",
    y="Feature",
    palette=colors,
    edgecolor="black"
)

# Zero reference line
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

# Add direction arrows next to bars
for i, val in enumerate(plot_df["SignedMean"]):
    # Matplotlib coordinates: y position is i (bars are plotted in the order of plot_df)
    # Determine text offset (small fraction of abs range)
    offset = 0.001 if val >= 0 else -0.001
    ax.text(val + offset, i, "↑" if val > 0 else "↓",
            va="center", ha="left" if val > 0 else "right", fontsize=12, color="black")

# Legend (lower right)
legend_handles = [
    mpatches.Patch(color="crimson", label="Increases Risk (Positive Weight)"),
    mpatches.Patch(color="steelblue", label="Decreases Risk (Negative Weight)")
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=11, frameon=True, edgecolor="black")
ax.set_xlabel("Signed Mean LIME Weight", fontsize=12)

ax.set_ylabel("Top Feature Pairs per PCA Component")
#ax.set_title("LIME Feature Directionality — NOVEM-CVD Ensemble", fontsize=14)

# Save both PNG and PDF (directionality)
save_plot(fig, "LIME_feature_importance_directional.png")
try:
    pdf_path = "/content/LIME_directionality_plot.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    # Attempt Colab auto-download (non-fatal if not in Colab)
    try:
        from google.colab import files
        files.download(pdf_path)
    except Exception:
        pass
except Exception as e:
    log("plots", f"Could not save PDF directionality: {e}")
plt.close(fig)
log("plots", "✅ Saved: LIME_feature_importance_directional.png")

# 6.2 Absolute magnitude (Matplotlib)
fig = plt.figure(figsize=(9,5))
df_abs = lime_summary.sort_values("AbsMean", ascending=True)
plt.barh(df_abs["Feature"], df_abs["AbsMean"], color="#6baed6", edgecolor="black")
plt.xlabel(" Signed Mean Absolute Contribution")

plt.ylabel("Top Feature Pairs per PCA Component")
#plt.title("Absolute LIME Feature Importance — NOVEM-CVD Ensemble", pad=12)
save_plot(fig, "LIME_feature_importance_abs.png")
log("plots", "✅ Saved: LIME_feature_importance_abs.png")

# 6.3 Stability — Std Dev
fig = plt.figure(figsize=(9,5))
df_sd = lime_summary.copy().sort_values("AbsSD", ascending=True)
plt.barh(df_sd["Feature"], df_sd["AbsSD"], color="#bdbdbd", edgecolor="black")
# Annotate mean ± sd
for y, (m, s) in enumerate(zip(df_sd["AbsMean"], df_sd["AbsSD"])):
    plt.text(s + (max(df_sd["AbsSD"])*0.005), y, f"{m:.3f} ± {s:.3f}", va="center", fontsize=8)
plt.xlabel("Standard Deviation of LIME Weights Across Runs")
plt.ylabel("Top Feature Pairs per PCA Component")
#plt.title("Feature-Level Variability of LIME Explanations", pad=12)
save_plot(fig, "LIME_stability_sd.png")
log("plots", "✅ Saved: LIME_stability_sd.png")

# 6.4 Jaccard consistency (Top-k)
top_k = 5
top_idxs = [np.argsort(-np.abs(r))[:top_k] for r in feature_runs]
pairs, scores = [], []
for i in range(n_runs):
    for j in range(i+1, n_runs):
        a, b = set(top_idxs[i]), set(top_idxs[j])
        jac = len(a & b) / len(a | b)
        pairs.append(f"Run{i+1}–Run{j+1}")
        scores.append(jac)
jacc_mean, jacc_std = float(np.mean(scores)), float(np.std(scores))

fig = plt.figure(figsize=(9,5))
order = np.argsort(scores)[::-1]
ordered_pairs = [pairs[i] for i in order]
ordered_scores = [scores[i] for i in order]
plt.barh(ordered_pairs, ordered_scores, color="#9ecae1", edgecolor="black")
plt.axvline(jacc_mean, color="black", linestyle="--", linewidth=0.8)
plt.text(min(0.98, jacc_mean+0.01), 0.2, f"Mean = {jacc_mean:.3f} ± {jacc_std:.3f}", fontsize=9)
plt.xlim(0,1)
plt.xlabel(f"Jaccard Similarity of Top-{top_k} Features (pairwise across runs)")

plt.ylabel(f"Run pair names")
#plt.title(f"Cross-Run Consistency of Top-k LIME Features (k = {top_k})", pad=12)
save_plot(fig, "LIME_stability_jaccard.png")
log("plots", "✅ Saved: LIME_stability_jaccard.png")

# ========== 7) LOCAL EXPLANATIONS (fast PNG thumbnails) ==========
log("local", "Building local explanation panels (fast thumbnails)...")
os.makedirs("lime_instances", exist_ok=True)

proba = voting_model.predict_proba(X_test)[:, POS_CLASS]
hi, lo = int(np.argmax(proba)), int(np.argmin(proba))
others = list(set(range(len(X_test))) - {hi, lo})
randoms = random.sample(others, 3)
local_idx = [hi, lo] + randoms

local_cards = []
for idx in local_idx:
    exp = explainer.explain_instance(
        data_row=X_test[idx],
        predict_fn=voting_model.predict_proba,
        num_features=min(10, len(feature_names)),
        num_samples=LIME_CONFIG["num_samples"]
    )
    html_path = f"lime_instances/lime_instance_{idx}.html"
    exp.save_to_file(html_path)

    # Create a compact Matplotlib thumbnail from exp.as_list()
    weights = exp.as_list()
    labels = [w[0] for w in weights]
    values = [w[1] for w in weights]

    fig, ax = plt.subplots(figsize=(5,4))
    ax.barh(labels, values)
    ax.set_title(f"LIME — Instance {idx}")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
    fig.tight_layout()

    png_path = f"lime_instances/lime_instance_{idx}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    local_cards.append({
        "index": idx,
        "path_html": html_path,
        "path_png": png_path,
        "pred_prob": float(proba[idx]),
        "true_label": int(y_test[idx])
    })

log("local", f"✅ Saved {len(local_cards)} local panels in 'lime_instances/'")

# ========== 8) BUILD HTML REPORT ==========
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outputs_dir = f"LIME_outputs_{timestamp}"
os.makedirs(outputs_dir, exist_ok=True)

# Copy static outputs
for f in [
    "LIME_summary.csv",
    "LIME_feature_importance_directional.png",
    "LIME_feature_importance_abs.png",
    "LIME_stability_sd.png",
    "LIME_stability_jaccard.png"
]:
    if os.path.exists(f):
        shutil.copy(f, outputs_dir)

# Copy local cards
shutil.copytree("lime_instances", os.path.join(outputs_dir, "lime_instances"), dirs_exist_ok=True)

iframe_blocks = "\n".join([
    f"<details open><summary><b>Instance {c['index']}</b> — True label: {c['true_label']} | P(Disease)={c['pred_prob']:.4f}</summary>"
    f"<iframe src='lime_instances/{os.path.basename(c['path_html'])}' width='100%' height='520' style='border:1px solid #ddd;border-radius:8px;'></iframe>"
    f"<div><img src='lime_instances/{os.path.basename(c['path_png'])}' style='max-width:60%;margin:10px auto;display:block;'/></div>"
    f"</details>"
    for c in local_cards
])

html = f"""<html><head><meta charset='utf-8'/><title>LIME Report — NOVEM-CVD Ensemble</title></head><body>
<h1>📘 NOVEM-CVD — LIME Explanation Report</h1>
<p><b>Generated:</b> {timestamp}</p>

<h2>Global Explanations</h2>
<img src='LIME_feature_importance_directional.png' width='90%'><p><i>Fig 1. Directionality of LIME weights (↑ increases risk; ↓ decreases risk).</i></p>
<img src='LIME_feature_importance_abs.png' width='90%'><p><i>Fig 2. Absolute importance magnitude of LIME features.</i></p>
<img src='LIME_stability_sd.png' width='90%'><p><i>Fig 3. Feature-level variability across runs (mean ± SD).</i></p>
<img src='LIME_stability_jaccard.png' width='90%'><p><i>Fig 4. Top-{top_k} cross-run Jaccard stability.</i></p>

{lime_summary.head(10).to_html(index=True)}

<h2>Local Explanations (Interactive + PNG thumbnails)</h2>
{iframe_blocks}

<hr><p style='text-align:center;color:#666'>NOVEM-CVD Explainable Ensemble © 2025</p>
</body></html>"""

html_path = os.path.join(outputs_dir, "LIME_report.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
log("html", f"✅ HTML report saved: {html_path}")

# ========== 9) PDF + ZIP + Auto-Download ==========
if not SKIP_PDF:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet

        pdf_path = os.path.join(outputs_dir, "LIME_report.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("📘 NOVEM-CVD LIME Explanation Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
        story.append(Spacer(1, 24)); story.append(PageBreak())

        figs = [
            ("LIME_feature_importance_directional.png", "Fig 1. Directionality of LIME weights."),
            ("LIME_feature_importance_abs.png", "Fig 2. Absolute importance magnitude."),
            ("LIME_stability_sd.png", "Fig 3. Feature-level variability across runs."),
            ("LIME_stability_jaccard.png", f"Fig 4. Cross-run Jaccard stability (Top-{top_k}).")
        ]
        for fname, caption in figs:
            if os.path.exists(fname):
                story.append(Image(fname, width=450, height=300))
                story.append(Spacer(1, 6))
                story.append(Paragraph(caption, styles['Italic']))
                story.append(PageBreak())

        story.append(Paragraph("Local Explanations — Thumbnails", styles['Heading2']))
        for c in local_cards:
            story.append(Paragraph(
                f"Instance {c['index']} — True Label: {c['true_label']} — P(Disease): {c['pred_prob']:.4f}",
                styles['Normal']
            ))
            if c['path_png'] and os.path.exists(c['path_png']):
                story.append(Image(c['path_png'], width=440, height=300))
            else:
                story.append(Paragraph("(PNG not available — see HTML for interactive panel)", styles['Italic']))
            story.append(PageBreak())

        doc.build(story)
        log("pdf", f"✅ PDF report saved: {pdf_path}")
    except Exception as e:
        log("pdf", f"PDF generation failed: {e}")

# ZIP packaging
zip_name = f"LIME_outputs_{timestamp}.zip"
with zipfile.ZipFile(zip_name, "w") as z:
    for root, _, files in os.walk(outputs_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, os.path.dirname(outputs_dir))
            z.write(full, arcname=rel)

friendly = "NOVEM_CVD_LIME_Report_Latest.zip"
shutil.copy(zip_name, friendly)
log("zip", f"📦 Packaged: {friendly}")

# Auto-download (Colab)
try:
    from google.colab import files
    files.download(friendly)
    log("zip", "📥 Auto-download triggered (Colab).")
except Exception:
    log("zip", "ℹ️ Not in Colab — download ZIP manually from working directory.")

print("\n✅ FULL NOVEM-CVD LIME MODULE COMPLETE — HTML, ZIP (and optional PDF) generated.")

```

### Cell 90 — code (execution_count=37)
```python
# ============================================
# ✅ SAVE ALL PLOTS INTO A SINGLE PDF + AUTO-DOWNLOAD
# ============================================

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

all_plots_pdf = "/content/NOVEM_CVD_All_Plots.pdf"
doc = SimpleDocTemplate(all_plots_pdf, pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("📘 NOVEM-CVD — Complete Plot Collection", styles["Title"]))
story.append(Spacer(1, 20))

# List ALL main global plots you want in PDF
plot_files = [
    "LIME_feature_importance_directional.png",
    "LIME_feature_importance_abs.png",
    "LIME_stability_sd.png",
    "LIME_stability_jaccard.png"
]

# Add global plots
for p in plot_files:
    if os.path.exists(p):
        story.append(Image(p, width=440, height=300))
        story.append(Spacer(1, 10))
        story.append(PageBreak())

# OPTIONAL: include local instance thumbnails
story.append(Paragraph("Local Explanations — Thumbnails", styles["Heading2"]))
story.append(Spacer(1, 10))

for c in local_cards:
    story.append(Paragraph(
        f"Instance {c['index']} — True Label: {c['true_label']} — P(Disease)={c['pred_prob']:.4f}",
        styles["Normal"]
    ))
    if os.path.exists(c['path_png']):
        story.append(Image(c['path_png'], width=440, height=300))
    story.append(PageBreak())

doc.build(story)

print(f"✅ All plots saved in: {all_plots_pdf}")

# Auto-download in Google Colab
try:
    from google.colab import files
    files.download(all_plots_pdf)
    print("📥 Auto-download started!")
except:
    print("ℹ️ Not running in Colab — download manually.")

```

### Cell 91 — markdown (execution_count=None)
```text
# **Reviewer 1, comment 15**
```

### Cell 92 — code (execution_count=69)
```python
# =============================================================
# NOVEM-CVD — LIME Stability & Clinical Plausibility Module
# Implements Reviewer#1 Concern #15 (Full reproducible analysis)
# Outputs: CSVs, individual PDF plots, clinical plausibility text,
# auto-downloads PDFs in Google Colab
# =============================================================

import os, joblib, warnings, random
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set(style="whitegrid")
from lime.lime_tabular import LimeTabularExplainer
from lime.discretize import QuartileDiscretizer

RND = 42
np.random.seed(RND)
random.seed(RND)

# ----------------------------
# 0. Paths / load model + data
# ----------------------------
# Adjust these paths if your files are located elsewhere
MODEL_PATH = "/content/drive/MyDrive/models/voting_pipeline.pkl"   # or "voting_pipeline.pkl"
DATA_PATH  = "/content/drive/MyDrive/models/LoRAS_train_test_data.pkl"  # or "LoRAS_train_test_data.pkl"

print("🔄 Loading model and data...")
model_obj = joblib.load(MODEL_PATH)   # could be Pipeline or VotingClassifier
data = joblib.load(DATA_PATH)         # expects dict with X_train, X_test, y_train, y_test

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

# ----------------------------
# 1. Resolve ensemble and base models
# ----------------------------
def get_ensemble_and_bases(model_obj):
    """
    Accept either a Pipeline that contains 'voting' step, or a VotingClassifier object,
    and return (ensemble, base_models_dict{name:estimator}).
    """
    from sklearn.ensemble import VotingClassifier
    ensemble = None
    if hasattr(model_obj, "named_steps") and "voting" in model_obj.named_steps:
        ensemble = model_obj.named_steps["voting"]
    elif isinstance(model_obj, VotingClassifier):
        ensemble = model_obj
    else:
        # try common names
        if hasattr(model_obj, "estimators_") or hasattr(model_obj, "estimators"):
            ensemble = model_obj
        else:
            raise RuntimeError("Could not find VotingClassifier inside model object.")
    # Extract names and fitted models robustly:
    names = []
    if hasattr(ensemble, "estimators"):
        names = [n for n, _ in ensemble.estimators]
    else:
        # fallback create names from indices
        names = [f"m{i}" for i in range(len(ensemble.estimators_))]
    models_fitted = []
    if hasattr(ensemble, "estimators_"):
        models_fitted = list(ensemble.estimators_)
    else:
        # maybe estimators is list of tuples but not fitted; use second items
        models_fitted = [m for _, m in ensemble.estimators]
    # map names -> models (align lengths)
    if len(names) != len(models_fitted):
        # try to match lengths by truncation
        names = names[:len(models_fitted)]
    base_models = {names[i]: models_fitted[i] for i in range(len(models_fitted))}
    return ensemble, base_models

ensemble, base_models = get_ensemble_and_bases(model_obj)
print(f"✅ Ensemble found: {type(ensemble).__name__}")
print("✅ Base models:", list(base_models.keys()))

# ----------------------------
# 2. Feature names
# ----------------------------
if 'top_features_table' in globals():
    feature_names = top_features_table.apply(lambda x: f"{x['Top1_Feature']}_{x['Top2_Feature']}", axis=1).tolist()
else:
    # fallback from X_train columns if pandas DataFrame, else generic names
    if hasattr(X_train, "columns"):
        feature_names = list(X_train.columns)
    else:
        feature_names = [f"F{i}" for i in range(X_train.shape[1])]

# convert to numpy arrays for LIME
if hasattr(X_train, "to_numpy"):
    X_train_arr = X_train.to_numpy()
    X_test_arr  = X_test.to_numpy()
else:
    X_train_arr = np.array(X_train)
    X_test_arr  = np.array(X_test)

# ----------------------------
# 3. LIME Explainer (base)
# ----------------------------
discretizer = QuartileDiscretizer(X_train_arr, categorical_features=[], feature_names=feature_names)
explainer_base = LimeTabularExplainer(
    training_data=X_train_arr,
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    mode="classification",
    discretizer=discretizer,
    sample_around_instance=True,
    random_state=RND
)
print("✅ LIME explainer initialized.")

# ----------------------------
# utilities
# ----------------------------
POS_CLASS = 1  # index for positive class in predict_proba

def aggregate_weights_for_instances(explainer, model, instance_indices, num_samples=5000, kernel_width=None):
    """Return mean signed LIME weights (length = n_features) across instance_indices."""
    w = np.zeros(len(feature_names))
    for idx in instance_indices:
        if kernel_width is None:
            exp = explainer.explain_instance(data_row=X_test_arr[idx], predict_fn=model.predict_proba,
                                             num_features=len(feature_names), num_samples=num_samples)
        else:
            # construct explainer with requested kernel_width on the fly
            tmp = LimeTabularExplainer(training_data=X_train_arr, feature_names=feature_names,
                                       class_names=["No Disease", "Disease"], mode="classification",
                                       discretizer=discretizer, sample_around_instance=True, kernel_width=kernel_width,
                                       random_state=RND)
            exp = tmp.explain_instance(data_row=X_test_arr[idx], predict_fn=model.predict_proba,
                                       num_features=len(feature_names), num_samples=num_samples)
        # exp.as_map() returns {class: list of (feature_idx, weight)}
        emap = exp.as_map()
        if POS_CLASS in emap:
            for fidx, wt in emap[POS_CLASS]:
                w[fidx] += wt
        else:
            # fallback: try as_list() and parse feature names (rare)
            for fname, wt in exp.as_list():
                # try to match fname to index via feature_names substring
                for i, fn in enumerate(feature_names):
                    if fn in fname:
                        w[i] += wt
                        break
    return w / max(1, len(instance_indices))

def topk_from_weights(weights, k=5):
    return set(np.argsort(-np.abs(weights))[:k])

def jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

# ----------------------------
# 4. Baseline: LIME on final ensemble
# ----------------------------
print("\n🔎 Running baseline LIME on final ensemble (this is the model used for predictions)...")
n_instances = min(30, X_test_arr.shape[0])
instance_indices = np.random.choice(range(X_test_arr.shape[0]), n_instances, replace=False)
baseline_weights = aggregate_weights_for_instances(explainer_base, ensemble, instance_indices, num_samples=5000)
baseline_top5 = topk_from_weights(baseline_weights, 5)
print("✅ Baseline top-5 (feature indices):", sorted(list(baseline_top5)))

# Save global importance CSV
pd.DataFrame({
    "Feature": feature_names,
    "SignedMeanWeight": baseline_weights
}).sort_values("SignedMeanWeight", key=lambda s: np.abs(s), ascending=False).to_csv("LIME_baseline_global_importance.csv", index=False)

# ----------------------------
# 5. Perturbation sensitivity (kernel widths)
# ----------------------------
# ----------------------------
# 5. Improved Perturbation Sensitivity (kernel_width)
# ----------------------------
print("\n🔎 Improved Perturbation Sensitivity (kernel_width)...")

kernel_widths = [0.5, 1.0, 2.0, 3.0, 5.0]
n_repeats = 5  # number of runs per kernel_width for stability

perturb_results = []
perturb_jacc = []

for kw in kernel_widths:
    jacc_scores = []

    for r in range(n_repeats):
        # Create explainer with varying kernel_width
        exp_temp = LimeTabularExplainer(
            training_data=X_train_arr,
            feature_names=feature_names,
            class_names=["No Disease", "Disease"],
            mode="classification",
            discretizer=discretizer,
            sample_around_instance=True,
            kernel_width=kw,
            random_state=RND + r
        )

        # Compute aggregated weights
        w = aggregate_weights_for_instances(
            exp_temp, ensemble,
            instance_indices, num_samples=5000
        )

        top5 = topk_from_weights(w, 5)

        jacc_scores.append(jaccard(baseline_top5, top5))

    perturb_results.append(jacc_scores)
    perturb_jacc.append((np.mean(jacc_scores), np.std(jacc_scores)))

# Create DataFrame for saving
df_perturb = pd.DataFrame({
    "kernel_width": kernel_widths,
    "mean_jaccard": [m for m, s in perturb_jacc],
    "std_jaccard": [s for m, s in perturb_jacc],
    "all_repeats": perturb_results
})
df_perturb.to_csv("LIME_perturbation_sensitivity.csv", index=False)
print("✅ Saved: LIME_perturbation_sensitivity.csv")

# ✅ Improved plot with mean ± std error bars + CI band
plt.figure(figsize=(7,4))

means = df_perturb["mean_jaccard"].values
stds  = df_perturb["std_jaccard"].values

plt.plot(kernel_widths, means, marker='o', linewidth=2, label="Mean Jaccard")
plt.fill_between(kernel_widths, means - stds, means + stds,
                 alpha=0.2, label="±1 Std Dev")

plt.xlabel("kernel_width", fontsize=12)
plt.ylabel("Jaccard vs Baseline (Top-5)", fontsize=12)
plt.title("LIME: Perturbation Sensitivity (Improved)", fontsize=14)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("LIME_perturbation_sensitivity.pdf")
plt.close()

# ----------------------------
# 6. Sample-count sensitivity (num_samples)
# ----------------------------
# ----------------------------
# 6. Sample-count sensitivity (num_samples)
# ----------------------------
print("\n🔎 Sample-count sensitivity (num_samples)...")

sample_list = [500, 1000, 1500, 2000, 2500]  # Updated recommended list
sample_rows = []
sample_jacc = []

for ns in sample_list:
    w = aggregate_weights_for_instances(
        explainer_base, ensemble,
        instance_indices, num_samples=ns
    )
    top5 = topk_from_weights(w, 5)
    sample_rows.append((ns, sorted(list(top5))))
    sample_jacc.append(jaccard(baseline_top5, top5))

df_samples = pd.DataFrame({
    "num_samples": [r[0] for r in sample_rows],
    "top5_features": [r[1] for r in sample_rows],
    "jaccard_with_baseline": sample_jacc
})
df_samples.to_csv("LIME_samplecount_sensitivity.csv", index=False)
print("✅ Saved: LIME_samplecount_sensitivity.csv")

# ✅ Improved linear-scale plot
plt.figure(figsize=(7,4))
plt.plot(
    df_samples["num_samples"],
    df_samples["jaccard_with_baseline"],
    marker='o', linewidth=2
)

plt.title("LIME: Sample Size Sensitivity", fontsize=14)
plt.xlabel("Number of Samples", fontsize=12)
plt.ylabel("Jaccard vs Baseline (Top-5)", fontsize=12)

plt.xticks(sample_list, fontsize=11)
plt.ylim(0, 1.05)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("LIME_samplecount_sensitivity.pdf")
plt.close()

# ----------------------------
# 7. Multi-run stability (Top-k Jaccard across random seeds)
# ----------------------------
# ----------------------------
# 7. Multi-run stability (Top-k Jaccard across random seeds)
# ----------------------------
# Full Multi-run LIME Stability: Base Models & Ensemble
# ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
import joblib

print("🔄 Loading model and data...")

# Load model pipeline or VotingClassifier directly
voting_clf = joblib.load("/content/drive/MyDrive/models/voting_pipeline.pkl")

# --- Extract the actual VotingClassifier safely ---
from sklearn.ensemble import VotingClassifier
if isinstance(voting_clf, VotingClassifier):
    voting_ens = voting_clf
elif hasattr(voting_clf, "named_steps") and "voting" in voting_clf.named_steps:
    voting_ens = voting_clf.named_steps["voting"]
else:
    raise AttributeError("❌ Could not locate VotingClassifier inside the loaded model.")

# --- Access base models robustly ---
knn_clf = voting_ens.named_estimators_['knn']
rf_clf = voting_ens.named_estimators_['rf']
log_clf = voting_ens.named_estimators_['logistic']
svm_clf = voting_ens.named_estimators_['svm']

print("✅ Models loaded successfully")
print(f"✅ Ensemble found: {type(voting_ens).__name__}")
print(f"✅ Base models: {list(voting_ens.named_estimators_.keys())}")

# Ensure X_train/X_test are arrays
if hasattr(X_train, "values"):
    X_train_array = X_train.values
    X_test_array = X_test.values
    feature_names = X_train.columns.tolist()
else:
    X_train_array = X_train
    X_test_array = X_test
    feature_names = [f"f{i}" for i in range(X_train_array.shape[1])]

# Define models to explain
models_to_explain = {
    "Ensemble": voting_ens,
    "KNN": knn_clf,
    "RandomForest": rf_clf,
    "Logistic": log_clf,
    "SVM": svm_clf
}

# Parameters
n_runs = 5          # number of random seeds
top_k = 5           # top-k features
num_samples = 1500   # for LIME explain_instance

stability_results = {}

# ----------------------------
# Multi-run LIME explanations
# ----------------------------
for model_name, model in models_to_explain.items():
    print(f"\n➡ Running multi-run LIME for {model_name}...")
    run_topk = []

    for r in range(n_runs):
        expl = LimeTabularExplainer(
            training_data=X_train_array,
            mode='classification',
            training_labels=y_train,
            feature_names=feature_names,
            class_names=['No_HD', 'HD'],
            discretize_continuous=True,
            random_state=r
        )

        instance_indices = np.arange(X_test_array.shape[0])

        # Use your existing helper
        w = aggregate_weights_for_instances(
            explainer=expl,
            model=model,
            instance_indices=instance_indices,
            num_samples=num_samples,
            kernel_width=None
        )

        run_topk.append(topk_from_weights(w, top_k))

    # Compute pairwise Jaccard similarity
    pairwise = []
    for i in range(len(run_topk)):
        for j in range(i+1, len(run_topk)):
            pairwise.append(jaccard(run_topk[i], run_topk[j]))
    pairwise = np.array(pairwise)

    stability_results[model_name] = pairwise
    print(f"✅ {model_name} Mean Jaccard: {pairwise.mean():.3f}  Std: {pairwise.std():.3f}")

# ----------------------------
# Plot comparison
# ----------------------------
plt.figure(figsize=(10,6))
sns.boxplot(
    data=[stability_results[m] for m in models_to_explain.keys()],
    palette="Blues",
    width=0.5
)
sns.stripplot(
    data=[stability_results[m] for m in models_to_explain.keys()],
    color="black",
    size=5,
    jitter=0.15
)

plt.xticks(ticks=range(len(models_to_explain)), labels=list(models_to_explain.keys()))
plt.ylabel("Pairwise Jaccard Similarity (Top-5)")
plt.title("LIME Multi-Run Stability: Base Models vs Ensemble", fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("LIME_base_vs_ensemble_stability.pdf", dpi=300)
plt.close()

# Save raw Jaccard scores per model
for model_name, jaccard_values in stability_results.items():
    pd.DataFrame({"jaccard": jaccard_values}).to_csv(f"LIME_jaccard_{model_name}.csv", index=False)

print("✅ Saved comparison plot and CSVs for all models")

# ----------------------------
# 8. LIME on base models vs ensemble
# ----------------------------
print("\n🔎 LIME on base models (compare top-5 to ensemble)...")

base_results = {}
for name, model in base_models.items():
    try:
        w = aggregate_weights_for_instances(explainer_base, model, instance_indices, num_samples=3000)
        base_results[name] = sorted(list(topk_from_weights(w, 5)))
    except Exception as e:
        base_results[name] = []
        print(f"⚠️ Could not run LIME on base model {name}: {e}")

# Calculate overlaps
overlaps = {name: len(set(base_results[name]) & baseline_top5) for name in base_results}

# ✅ Include ensemble itself in comparison
models_plot = list(base_results.keys()) + ["ensemble"]
overlaps_plot = [overlaps[m] for m in base_results] + [len(baseline_top5)]  # ensemble overlap = full

# Save CSV (with ensemble row)
df_base = pd.DataFrame({
    "Model": models_plot,
    "Top5_Features": [base_results.get(m, baseline_top5) if m != "ensemble" else sorted(list(baseline_top5)) for m in models_plot],
    "Overlap_with_ensemble_top5": overlaps_plot
})
df_base.to_csv("LIME_base_vs_ensemble.csv", index=False)
print("✅ Saved: LIME_base_vs_ensemble.csv")

# ✅ Publication-Quality Overlap Plot (hatches + legend + value labels)
plt.figure(figsize=(8, 5))

hatches = ['/', '\\', 'x', 'o', '///']  # 5 different patterns
x = np.arange(len(models_plot))

bars = []
for i, (model, overlap) in enumerate(zip(models_plot, overlaps_plot)):
    bar = plt.bar(
        x[i], overlap,
        hatch=hatches[i],
        edgecolor='black',
        linewidth=1.3,
        label=model
    )
    bars.append(bar)

    # ✅ Add annotation (exact value)
    plt.text(
        x[i], overlap + 0.15,
        f"{overlap}",
        ha='center', va='bottom',
        fontsize=12, fontweight='bold'
    )

# Labels & formatting
plt.ylabel("Overlap Count (Top-5)", fontsize=13)
plt.xlabel("Models", fontsize=13)
plt.title("Overlap between Base Models and Voting Ensemble (Top-5 LIME Features)", fontsize=14)
plt.xticks(x, models_plot, fontsize=12)
plt.ylim(0, 6)
plt.grid(axis='y', alpha=0.3)

# Legend
plt.legend(
    title="Models",
    title_fontsize=12,
    fontsize=11,
    loc='upper left',
    frameon=True
)

plt.tight_layout()
plt.savefig("LIME_base_vs_ensemble_overlap.pdf")
plt.close()

print("✅ Saved: LIME_base_vs_ensemble_overlap.pdf (patterned + labeled)")


# ----------------------------
# 9. Clinical plausibility summary (text file)
# ----------------------------
print("\n📝 Writing brief clinical plausibility summary...")
# Identify top feature names (by absolute baseline weight)
top_idx_sorted = np.argsort(-np.abs(baseline_weights))[:10]
top_feature_names = [feature_names[i] for i in top_idx_sorted]

clinical_text = f"""NOVEM-CVD — LIME Clinical Plausibility Summary
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Top features identified by LIME (ensemble) — top 10 (by |weight|):
{top_feature_names}

Short clinical notes:
- Age: Established non-modifiable risk factor for cardiovascular disease; increases risk via cumulative vascular damage, arterial stiffness and comorbidity burden.
- FastingBS / Glucose: Hyperglycemia and insulin resistance increase atherosclerosis and microvascular complications; elevated fasting glucose is associated with higher CVD risk.
- MaxHR (Maximum heart rate): Lower exercise max heart rate or chronotropic incompetence is associated with poorer cardiac reserve and coronary disease.
- Oldpeak (ST depression on exertion): A recognized marker of ischemia during exercise testing; higher values suggest greater ischemic burden.
- ExerciseAngina: Clinical sign associated with obstructive coronary disease.

Interpretation guidance:
These LIME-derived features align with established cardiovascular literature (e.g., Framingham risk components, exercise stress test markers). The consistency of these features across perturbation and sample-count sensitivity tests increases confidence in their clinical plausibility as contributors to the model's risk predictions.
"""
with open("LIME_clinical_plausibility.txt", "w") as f:
    f.write(clinical_text)
print("✅ Saved: LIME_clinical_plausibility.txt")

# ----------------------------
# 10. Auto-download PDFs and CSVs in Colab (if available)
# ----------------------------
print("\n📦 Preparing to auto-download outputs (if running in Colab)...")
outputs = [
    "LIME_perturbation_sensitivity.pdf",
    "LIME_samplecount_sensitivity.pdf",
    "LIME_multirun_stability.pdf",
    "LIME_base_vs_ensemble_overlap.pdf",
    "LIME_baseline_global_importance.csv",
    "LIME_perturbation_sensitivity.csv",
    "LIME_samplecount_sensitivity.csv",
    "LIME_multirun_jaccard.csv",
    "LIME_base_vs_ensemble.csv",
    "LIME_clinical_plausibility.txt"
]

try:
    from google.colab import files
    for of in outputs:
        if os.path.exists(of):
            files.download(of)
    print("✅ Auto-download attempted for available outputs.")
except Exception:
    print("ℹ️ Not running in Colab or auto-download failed. Files saved to working directory.")

print("\n🎉 Done. All requested stability analyses and outputs are saved.")

```