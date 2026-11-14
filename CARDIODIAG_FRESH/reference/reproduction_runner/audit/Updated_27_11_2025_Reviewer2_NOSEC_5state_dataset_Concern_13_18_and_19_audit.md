# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\five_states_results\Five States Results\Updated_27_11_2025_Reviewer2_NOSEC_5state_dataset_Concern_13_18_and_19.ipynb`
- SHA-256: `cc9102803a400d8e96f6aac46a2a51822f692c495f8c9e90eea4ea1ec43670f2`
- Cells: 146
- Saved outputs: 188

## Imports
- `catboost`
- `google.colab`
- `imblearn.over_sampling`
- `lightgbm`
- `loras`
- `matplotlib.pyplot`
- `numpy`
- `pandas`
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
- **credential_or_mount**: cells [1, 76, 85, 93, 112, 113, 121, 124, 135, 139, 140, 142]
- **notebook_shell**: cells [7, 17, 67, 103]

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
# **Box Plot on Original Data**
```

### Cell 7 — code (execution_count=None)
```python
!pip install pandas matplotlib scipy
from scipy.stats import skew
# Select numerical columns from the CORRECT DataFrame (e.g., data)
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Calculate subplot layout
num_cols = len(numerical_columns)
nrows = (num_cols + 2) // 3
ncols = min(num_cols, 3)

# Create subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
axes = axes.flatten()

# Loop through columns in data (not X)
for i, column in enumerate(numerical_columns):
    column_skewness = skew(data[column], axis=0, nan_policy='omit')  # Use data
    data.boxplot(column=column, ax=axes[i])  # Use data
    axes[i].set_title(f'{column}, skewness: {column_skewness:.2f}')

# Hide empty subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

# Adjust layout
fig.suptitle('Boxplots before Outliers Handling', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
```

### Cell 8 — markdown (execution_count=None)
```text
# **Anamoly/ Outllier Dection by Percentile Method**
```

### Cell 9 — code (execution_count=None)
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

### Cell 10 — markdown (execution_count=None)
```text
# **Dealing Outliers with Data Imputation Methods **
```

### Cell 11 — code (execution_count=None)
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

### Cell 12 — code (execution_count=None)
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

### Cell 13 — code (execution_count=None)
```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# Select numerical columns for boxplot, excluding 'HeartDisease'
numerical_columns = df_imputed.select_dtypes(include=['float64', 'int64']).columns
# numerical_columns = numerical_columns[numerical_columns != 'HeartDisease']  # Exclude 'HeartDisease'

# Calculate the number of rows and columns for subplots
# based on the number of numerical columns
num_cols = len(numerical_columns)
nrows = (num_cols + 2) // 3  # Calculate rows, ensuring at least 2 rows
ncols = min(num_cols, 3)    # Limit columns to 3

# Set up the layout for boxplots (dynamically determined)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
axes = axes.flatten()

# Loop through each column to create a boxplot and compute skewness
for i, column in enumerate(numerical_columns):
    # Calculate skewness for the current column
    column_skewness = skew(df_imputed[column], axis=0, nan_policy='omit')

    # Create a boxplot for the column using df_imputed
    df_imputed.boxplot(column=column, ax=axes[i]) # Changed to df_imputed

    # Set title with skewness value
    axes[i].set_title(f'{column}, skewness: {column_skewness:.2f}')

# Hide any empty subplots (if necessary)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

# Add a main title
fig.suptitle('Boxplots for each variable after Anomalies Removal', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust top so the title does not overlap with plots
plt.show()
```

### Cell 14 — code (execution_count=None)
```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

data_original=X
# Select numerical columns for boxplot, excluding 'HeartDisease'
numerical_columns = data_original.select_dtypes(include=['float64', 'int64']).columns
numerical_columns = numerical_columns[numerical_columns != 'HeartDisease']  # Exclude 'HeartDisease'

# Calculate the number of rows and columns for subplots
num_cols = len(numerical_columns)
nrows = (num_cols + 2) // 3  # Calculate rows, ensuring at least 2 rows
ncols = min(num_cols, 3)    # Limit columns to 3

# Set up the layout for boxplots (side-by-side comparison)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))
axes = axes.flatten()

# Loop through each column to create side-by-side boxplots and compute skewness
for i, column in enumerate(numerical_columns):
    # Calculate skewness for the current column in both datasets
    skew_original = skew(data_original[column], axis=0, nan_policy='omit')
    skew_no_outliers = skew(df_imputed[column], axis=0, nan_policy='omit')

    # Create a boxplot for both the original and outlier-removed data
    axes[i].boxplot([data_original[column].dropna(), df_imputed[column].dropna()],
                    labels=['Original', 'No Outliers'])

    # Set title with skewness values for both datasets
    axes[i].set_title(f'{column}\nOriginal skew: {skew_original:.4f}, No outliers skew: {skew_no_outliers:.4f}')

# Hide any empty subplots (if necessary)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

# Add a main title
fig.suptitle('Boxplots Comparison (Original vs. After Outliers Removal)', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90)  # Adjust top so the title does not overlap with plots
plt.show()

```

### Cell 15 — code (execution_count=None)
```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# Select ONLY true numerical features (exclude categoricals)
numerical_columns = ['Age', 'Cholesterol', 'RestingBP', 'MaxHR']  # Example list

# Define layout
ncols = 2  # Fewer columns for wider plots
nrows = (len(numerical_columns) + 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4*nrows))
axes = axes.flatten()

# Custom colors for clarity
colors = {'Original': 'lightblue', 'Processed': 'salmon'}

for i, col in enumerate(numerical_columns):
    # Calculate skewness
    skew_orig = skew(data_original[col].dropna())
    skew_proc = skew(df_imputed[col].dropna())

    # Plot boxplots with distinct colors
    bp = axes[i].boxplot(
        [data_original[col].dropna(), df_imputed[col].dropna()],
        labels=['Original', 'Processed'],
        patch_artist=True,
        widths=0.6
    )

    # Color boxes
    for patch, color in zip(bp['boxes'], [colors['Original'], colors['Processed']]):
        patch.set_facecolor(color)

    # Add skewness as text annotation
    axes[i].text(
        0.95, 0.85,
        f"Skew:\nOrig: {skew_orig:.2f}\nProc: {skew_proc:.2f}",
        transform=axes[i].transAxes,
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    axes[i].set_title(col, fontweight='bold')
    axes[i].grid(alpha=0.3)

# Hide unused subplots
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

# Add legend
fig.legend(
    [bp['boxes'][0], bp['boxes'][1]],
    ['Original', 'Processed'],
    loc='lower right',
    bbox_to_anchor=(0.92, 0.05)
)

plt.suptitle('Distribution Comparison: Original vs. Processed Data (Outliers Removed)', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell 16 — markdown (execution_count=None)
```text


# Step 2: Apply Oversampling Technoques/ Data Balancing Technique LoRAS

```

### Cell 17 — code (execution_count=None)
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

### Cell 18 — code (execution_count=None)
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

### Cell 19 — markdown (execution_count=None)
```text
# **PCA to 9**
```

### Cell 20 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(LoRAS_features)
```

### Cell 21 — code (execution_count=None)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, LoRAS_labels, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 22 — markdown (execution_count=None)
```text
**Algorithms hyper parameter-tuning**
```

### Cell 23 — markdown (execution_count=None)
```text
# **DT**
```

### Cell 24 — code (execution_count=None)
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

### Cell 25 — code (execution_count=None)
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

### Cell 26 — code (execution_count=None)
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

### Cell 27 — code (execution_count=None)
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Ensure the model is fitted
# If this step is missing, it will cause the NotFittedError
grid_search_tree.fit(X_train, y_train)
best_tree = grid_search_tree.best_estimator_

# Feature names and target names
f_n = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
       "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", 'ST_Slope']
t_n = ["0", "1"]

# Create a figure for the plot
fig = plt.figure(figsize=(20, 10), dpi=100)

# Plot the decision tree
plot_tree(best_tree, feature_names=f_n, class_names=t_n, filled=True)

# Show the plot
plt.show()

```

### Cell 28 — code (execution_count=None)
```python
################################
# mean ± SD for ALL Metrics (Decision Tree)
################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import make_scorer
import numpy as np

# -------------------------
# 1. Decision Tree Grid Search
# -------------------------
param_grid_tree = {
    'max_depth': [2, 4, 5, None],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
best_tree = grid_search_tree.best_estimator_

print("Best Parameters:", best_tree)

# -------------------------
# 2. Mean ± SD for ALL Metrics Using CV (Reviewer #2 requirement)
# -------------------------
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(best_tree, X_train, y_train, cv=5, scoring=scoring)

print("\n=== Decision Tree Performance (Mean ± SD over 5-Fold CV) ===")
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")

# Now you have:
# Accuracy (mean ± SD)
# Precision (mean ± SD)
# Recall (mean ± SD)
# F1-score (mean ± SD)
# ROC-AUC (mean ± SD)
# -------------------------
# 3. Train–test evaluation (OPTIONAL — confusion matrix etc.)
# -------------------------
best_tree.fit(X_train, y_train)
y_pred_tree = best_tree.predict(X_test)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

test_acc = accuracy_score(y_test, y_pred_tree)
test_prec = precision_score(y_test, y_pred_tree)
test_rec = recall_score(y_test, y_pred_tree)
test_f1 = f1_score(y_test, y_pred_tree)

print("\n=== Test Set Metrics (Single Split) ===")
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall    : {test_rec:.4f}")
print(f"F1-score  : {test_f1:.4f}")

# AUC-ROC
if hasattr(best_tree, "predict_proba"):
    y_prob = best_tree.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_test = auc(fpr, tpr)
    print(f"AUC-ROC   : {auc_test:.4f}")

```

### Cell 29 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import numpy as np

# Example metrics with mean ± SD from cross_validate
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']

means = [
    cv_results['test_accuracy'].mean(),
    cv_results['test_precision'].mean(),
    cv_results['test_recall'].mean(),
    cv_results['test_f1'].mean(),
    cv_results['test_roc_auc'].mean()
]

stds = [
    cv_results['test_accuracy'].std(),
    cv_results['test_precision'].std(),
    cv_results['test_recall'].std(),
    cv_results['test_f1'].std(),
    cv_results['test_roc_auc'].std()
]

x = np.arange(len(metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=6, linewidth=2, markersize=8)

plt.xticks(x, metrics, fontsize=12)
plt.ylabel('Score', fontsize=14)
plt.title('Decision Tree Classifier Performance (Mean ± SD over 5-fold CV)', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 1)   # because metrics are between 0 and 1
plt.show()

```

### Cell 30 — markdown (execution_count=None)
```text
# **Gradient Boosting**
```

### Cell 31 — code (execution_count=None)
```python
'''# B Gradient Boosting
param_grid_gradien = {
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [20, 50, 100, 150]
}

# B Gradient Boosting Grid Search (Base 2 Classifier)
gradien = GradientBoostingClassifier()
grid_search_gradien = GridSearchCV(gradien, param_grid_gradien, cv=5, scoring='accuracy')
grid_search_gradien.fit(X_train, y_train)
best_gradien = grid_search_gradien.best_estimator_
# Output the best parameters
print("Best Parameters of Gradient Boosting Grid Search (Base 2 Classifier):", best_gradien)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_gradien.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_gradien = best_gradien.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_gradien:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_gradien = cross_val_score(best_gradien, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_gradien = cv_scores_gradien.mean()
print(f"Validation Accuracy of Decision gradien: {validation_accuracy_gradien:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_gradien = best_gradien.predict(X_test)

# 4. Calculate other performance metrics
print("Confusion Matrix of gradien:")
print(confusion_matrix(y_test, y_pred_gradien))

print("Classification Report of gradien:")
print(classification_report(y_test, y_pred_gradien))

# Optional: If you want to calculate accuracy directly
test_accuracy_gradien = accuracy_score(y_test, y_pred_gradien)
print(f"Test Accuracy of gradien: {test_accuracy_gradien:.4f}")
'''
```

### Cell 32 — markdown (execution_count=None)
```text
# **Random Forest**


```

### Cell 33 — code (execution_count=None)
```python
'''# D Random Forest
param_grid_rf = {
    'n_estimators': [25, 50, 100],
    'max_depth': [5, None],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [5, 10, None],
    'criterion': ['gini', 'entropy']
}

# D Random Forest Grid Search (Base 3 Classifier)
rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
# Output the best parameters
print("Best Parameters of Random Forest Grid Search (Base 3 Classifier):", best_rf)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_rf.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_rf = best_rf.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_rf:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_rf = cv_scores_rf.mean()
print(f"Validation Accuracy of Decision rf: {validation_accuracy_rf:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_rf = best_rf.predict(X_test)

# 4. Calculate other performance metrics
print("Confusion Matrix of rf:")
print(confusion_matrix(y_test, y_pred_rf))

print("Classification Report of rf:")
print(classification_report(y_test, y_pred_rf))

# Optional: If you want to calculate accuracy directly
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Test Accuracy of rf: {test_accuracy_rf:.4f}")
'''
```

### Cell 34 — code (execution_count=None)
```python
'''import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

# Define and train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Extract a single tree from the random forest (for example, the first tree)
estimator = rf.estimators_[0]

# Define the feature names and target names
f_n = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
       "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", 'ST_Slope']
t_n = ["0", "1"]

# Create a figure for the plot
fig = plt.figure(figsize=(20, 10), dpi=100)

# Plot the decision tree
plot_tree(estimator, feature_names=f_n, class_names=t_n, filled=True, max_depth=3)  # Adjust max_depth for readability

# Show the plot
plt.show()
'''
```

### Cell 35 — markdown (execution_count=None)
```text
# **Logistic Regression**
```

### Cell 36 — code (execution_count=None)
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

### Cell 37 — code (execution_count=None)
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

### Cell 38 — code (execution_count=None)
```python
# ============================================================
# Logistic Regression (Meta Classifier) – FULL CODE
# With Grid Search + Train/Test Evaluation + Mean ± SD (CV)
# Reviewer #2 Concern #19 Fully Satisfied
# Variables made unique using "ys52_" prefix
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report, confusion_matrix
)

# ------------------------------------------------------------
# 1. Parameter Grid for Logistic Regression (Meta Classifier)
# ------------------------------------------------------------
ys52_param_grid = {
    'penalty': ['l1', 'l2', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']
}

ys52_logistic = LogisticRegression(max_iter=1000)

ys52_grid_search = GridSearchCV(
    ys52_logistic, ys52_param_grid, cv=5, scoring='accuracy'
)

ys52_grid_search.fit(X_train, y_train)

ys52_best_logistic = ys52_grid_search.best_estimator_

print("Best Parameters (ys52 Logistic Regression Meta Classifier):")
print(ys52_best_logistic)

# ------------------------------------------------------------
# 2. Fit on Training Data
# ------------------------------------------------------------
ys52_best_logistic.fit(X_train, y_train)

# Training Accuracy
ys52_train_accuracy = ys52_best_logistic.score(X_train, y_train)
print(f"\nTraining Accuracy_ys52: {ys52_train_accuracy:.4f}")

# ------------------------------------------------------------
# 3. Validation Accuracy using CV
# ------------------------------------------------------------
ys52_cv_scores = cross_val_score(
    ys52_best_logistic, X_train, y_train, cv=5, scoring='accuracy'
)

ys52_validation_accuracy = ys52_cv_scores.mean()
print(f"Validation Accuracy_ys52: {ys52_validation_accuracy:.4f}")

# ------------------------------------------------------------
# 4. Predictions on Test Set
# ------------------------------------------------------------
ys52_predictions = ys52_best_logistic.predict(X_test)

print("\nConfusion Matrix_ys52:")
print(confusion_matrix(y_test, ys52_predictions))

print("\nClassification Report_ys52:")
print(classification_report(y_test, ys52_predictions))

ys52_test_accuracy = accuracy_score(y_test, ys52_predictions)
print(f"Testing Accuracy_ys52: {ys52_test_accuracy:.4f}")

# ------------------------------------------------------------
# 5. Compute Test Metrics (Precision, Recall, F1, AUC)
# ------------------------------------------------------------
ys52_precision = precision_score(y_test, ys52_predictions)
ys52_recall = recall_score(y_test, ys52_predictions)
ys52_f1 = f1_score(y_test, ys52_predictions)

# AUC-ROC
if hasattr(ys52_best_logistic, "predict_proba"):
    ys52_probabilities = ys52_best_logistic.predict_proba(X_test)[:, 1]
    fpr_ys52, tpr_ys52, thresh_ys52 = roc_curve(y_test, ys52_probabilities)
    ys52_auc = auc(fpr_ys52, tpr_ys52)
else:
    ys52_auc = None

print(f"Precision_ys52: {ys52_precision:.4f}")
print(f"Recall_ys52: {ys52_recall:.4f}")
print(f"F1 Score_ys52: {ys52_f1:.4f}")
print(f"AUC-ROC_ys52: {ys52_auc:.4f}")

# ------------------------------------------------------------
# 6. Plot ROC Curve
# ------------------------------------------------------------
if ys52_auc is not None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys52, tpr_ys52, color='blue',
             label=f'Logistic Regression (AUC = {ys52_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve – Logistic Regression (ys52)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# ------------------------------------------------------------
# 7. Confusion Matrix Plot (Counts + Percentages)
# ------------------------------------------------------------
ys52_cm = confusion_matrix(y_test, ys52_predictions)
ys52_cm_percent = ys52_cm.astype('float') / ys52_cm.sum(axis=1)[:, np.newaxis] * 100

ys52_labels = np.array([
    [f"{count}\n({percent:.1f}%)" for count, percent in zip(r1, r2)]
    for r1, r2 in zip(ys52_cm, ys52_cm_percent)
])

plt.figure(figsize=(6, 5))
sns.heatmap(
    ys52_cm, annot=ys52_labels, fmt="", cmap="Blues",
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (ys52 Logistic Regression)")
plt.show()

# ------------------------------------------------------------
# 8. Mean ± SD for Logistic Regression (Reviewer #2 Concern #19)
# ------------------------------------------------------------
ys52_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys52_cv_results = cross_validate(
    ys52_best_logistic, X_train, y_train, cv=5, scoring=ys52_scoring
)

print("\n=== Logistic Regression (ys52) Mean ± SD over 5-Fold CV ===")
for metric in ys52_scoring.keys():
    scores = ys52_cv_results[f'test_{metric}']
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")

```

### Cell 39 — code (execution_count=None)
```python
# ============================================================
# Mean ± SD Visualization Plot for ys52 Logistic Regression
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# Extract metrics from CV results
ys52_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']

ys52_means = [
    ys52_cv_results['test_accuracy'].mean(),
    ys52_cv_results['test_precision'].mean(),
    ys52_cv_results['test_recall'].mean(),
    ys52_cv_results['test_f1'].mean(),
    ys52_cv_results['test_roc_auc'].mean()
]

ys52_stds = [
    ys52_cv_results['test_accuracy'].std(),
    ys52_cv_results['test_precision'].std(),
    ys52_cv_results['test_recall'].std(),
    ys52_cv_results['test_f1'].std(),
    ys52_cv_results['test_roc_auc'].std()
]

# X-axis values
x = np.arange(len(ys52_metrics))

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys52_means,
    yerr=ys52_stds,
    fmt='o-', capsize=6, linewidth=2, markersize=8, color='blue'
)

plt.xticks(x, ys52_metrics, fontsize=12)
plt.ylabel('Score', fontsize=13)
plt.ylim(0, 1)
plt.title('Mean ± SD of Logistic Regression (ys52) Performance Metrics', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

```

### Cell 40 — markdown (execution_count=None)
```text
# **Bayesian Networks**
```

### Cell 41 — code (execution_count=None)
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

### Cell 42 — code (execution_count=None)
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

### Cell 43 — code (execution_count=None)
```python
# ============================================================
# Naïve Bayes (Base Classifier) – FULL CODE
# With Grid Search + Train/Test Evaluation + Mean ± SD (CV)
# Reviewer #2 Concern #19 Fully Satisfied
# Variables made unique using "ys53_" prefix
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report, confusion_matrix
)

# ------------------------------------------------------------
# 1. Parameter Grid for Naïve Bayes
# ------------------------------------------------------------
ys53_param_grid = {
    'var_smoothing': np.logspace(0, -9, 100)
}

ys53_bayes = GaussianNB()

ys53_grid_search = GridSearchCV(
    ys53_bayes, ys53_param_grid, cv=5, scoring='accuracy'
)

ys53_grid_search.fit(X_train, y_train)

ys53_best_bayes = ys53_grid_search.best_estimator_

print("\nBest Parameters (ys53 Naïve Bayes Base Classifier):")
print(ys53_best_bayes)

# ------------------------------------------------------------
# 2. Fit Model on Training Data
# ------------------------------------------------------------
ys53_best_bayes.fit(X_train, y_train)

# Training Accuracy
ys53_train_accuracy = ys53_best_bayes.score(X_train, y_train)
print(f"\nTraining Accuracy_ys53: {ys53_train_accuracy:.4f}")

# ------------------------------------------------------------
# 3. Validation Accuracy via 5-fold CV
# ------------------------------------------------------------
ys53_cv_scores = cross_val_score(
    ys53_best_bayes, X_train, y_train, cv=5, scoring='accuracy'
)

print(f"Validation Accuracy_ys53: {ys53_cv_scores.mean():.4f}")

# ------------------------------------------------------------
# 4. Predictions on Test Set
# ------------------------------------------------------------
ys53_predictions = ys53_best_bayes.predict(X_test)

print("\nConfusion Matrix_ys53:")
print(confusion_matrix(y_test, ys53_predictions))

print("\nClassification Report_ys53:")
print(classification_report(y_test, ys53_predictions))

ys53_test_accuracy = accuracy_score(y_test, ys53_predictions)
print(f"Test Accuracy_ys53: {ys53_test_accuracy:.4f}")

# ------------------------------------------------------------
# 5. Test Metrics (Precision, Recall, F1, AUC)
# ------------------------------------------------------------
ys53_precision = precision_score(y_test, ys53_predictions)
ys53_recall = recall_score(y_test, ys53_predictions)
ys53_f1 = f1_score(y_test, ys53_predictions)

if hasattr(ys53_best_bayes, "predict_proba"):
    ys53_probabilities = ys53_best_bayes.predict_proba(X_test)[:, 1]
    fpr_ys53, tpr_ys53, _ = roc_curve(y_test, ys53_probabilities)
    ys53_auc = auc(fpr_ys53, tpr_ys53)
else:
    ys53_auc = None

print(f"Precision_ys53: {ys53_precision:.4f}")
print(f"Recall_ys53: {ys53_recall:.4f}")
print(f"F1 Score_ys53: {ys53_f1:.4f}")
print(f"AUC-ROC_ys53: {ys53_auc:.4f}")

# ------------------------------------------------------------
# 6. ROC Curve Plot
# ------------------------------------------------------------
if ys53_auc is not None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys53, tpr_ys53, color='blue',
             label=f'Naïve Bayes (AUC = {ys53_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve – Naïve Bayes (ys53)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# ------------------------------------------------------------
# 7. Confusion Matrix Plot (Counts + Percentages)
# ------------------------------------------------------------
ys53_cm = confusion_matrix(y_test, ys53_predictions)
ys53_cm_percent = ys53_cm.astype('float') / ys53_cm.sum(axis=1)[:, np.newaxis] * 100

ys53_labels = np.array([
    [f"{count}\n({percent:.1f}%)" for count, percent in zip(r1, r2)]
    for r1, r2 in zip(ys53_cm, ys53_cm_percent)
])

plt.figure(figsize=(6, 5))
sns.heatmap(
    ys53_cm, annot=ys53_labels, fmt="", cmap="Blues",
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (ys53 Naïve Bayes)")
plt.show()

# ------------------------------------------------------------
# 8. Mean ± SD using Cross-Validation (Reviewer #2 Concern #19)
# ------------------------------------------------------------
ys53_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys53_cv_results = cross_validate(
    ys53_best_bayes, X_train, y_train, cv=5, scoring=ys53_scoring
)

print("\n=== Naïve Bayes (ys53) Mean ± SD over 5-Fold CV ===")
for metric in ys53_scoring.keys():
    scores = ys53_cv_results[f'test_{metric}']
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")

```

### Cell 44 — code (execution_count=None)
```python
# ===========================================================
# Mean ± SD Visualization Plot for ys53 Naive Bayes Classifier
# ===========================================================

import matplotlib.pyplot as plt
import numpy as np

# Metrics to visualize
ys53_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']

# Extract means from CV results
ys53_means = [
    ys53_cv_results['test_accuracy'].mean(),
    ys53_cv_results['test_precision'].mean(),
    ys53_cv_results['test_recall'].mean(),
    ys53_cv_results['test_f1'].mean(),
    ys53_cv_results['test_roc_auc'].mean()
]

# Extract standard deviations
ys53_stds = [
    ys53_cv_results['test_accuracy'].std(),
    ys53_cv_results['test_precision'].std(),
    ys53_cv_results['test_recall'].std(),
    ys53_cv_results['test_f1'].std(),
    ys53_cv_results['test_roc_auc'].std()
]

# X-axis positions
x = np.arange(len(ys53_metrics))

# Plot (Error Bar Plot)
plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys53_means,
    yerr=ys53_stds,
    fmt='o-', markersize=8, capsize=6, linewidth=2,
    color='darkblue', ecolor='steelblue'
)

# Axis formatting
plt.xticks(x, ys53_metrics, fontsize=12)
plt.ylabel('Score', fontsize=13)
plt.title('Mean ± SD of Naive Bayes (ys53) Performance Metrics', fontsize=15)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

```

### Cell 45 — markdown (execution_count=None)
```text
# **AdaBoost**
```

### Cell 46 — code (execution_count=None)
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

### Cell 47 — code (execution_count=None)
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

### Cell 48 — code (execution_count=None)
```python
# ============================================================
# AdaBoost Classifier (ys55)
# Full Pipeline: Grid Search → Evaluation → ROC → CM → Mean±SD → Plot
# Reviewer #2 Concern #19 Fully Addressed
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# 1. Grid Search for AdaBoost (ys55)
# ------------------------------------------------------------

ys55_param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.1, 1.0, 1.1, 1.2]
}

ys55_ab = AdaBoostClassifier()

ys55_grid_search = GridSearchCV(
    ys55_ab, ys55_param_grid,
    cv=5, scoring='accuracy'
)

ys55_grid_search.fit(X_train, y_train)
ys55_best_ab = ys55_grid_search.best_estimator_

print("\nBest AdaBoost Parameters (ys55):")
print(ys55_best_ab)

# ------------------------------------------------------------
# 2. Fit Model
# ------------------------------------------------------------

ys55_best_ab.fit(X_train, y_train)

# Training accuracy
ys55_train_acc = ys55_best_ab.score(X_train, y_train)
print(f"\nTraining Accuracy_ys55: {ys55_train_acc:.4f}")

# 5-fold CV Validation Accuracy
ys55_val_acc = cross_val_score(
    ys55_best_ab, X_train, y_train,
    cv=5, scoring='accuracy'
).mean()
print(f"Validation Accuracy_ys55: {ys55_val_acc:.4f}")

# ------------------------------------------------------------
# 3. Predictions on Test Set
# ------------------------------------------------------------

ys55_pred = ys55_best_ab.predict(X_test)

print("\nConfusion Matrix_ys55:")
print(confusion_matrix(y_test, ys55_pred))

print("\nClassification Report_ys55:")
print(classification_report(y_test, ys55_pred))

ys55_test_acc = accuracy_score(y_test, ys55_pred)
print(f"Testing Accuracy_ys55: {ys55_test_acc:.4f}")

# ------------------------------------------------------------
# 4. Precision, Recall, F1, AUC
# ------------------------------------------------------------

ys55_prec = precision_score(y_test, ys55_pred)
ys55_rec = recall_score(y_test, ys55_pred)
ys55_f1 = f1_score(y_test, ys55_pred)

if hasattr(ys55_best_ab, "predict_proba"):
    ys55_prob = ys55_best_ab.predict_proba(X_test)[:, 1]
    fpr_ys55, tpr_ys55, _ = roc_curve(y_test, ys55_prob)
    ys55_auc = auc(fpr_ys55, tpr_ys55)
else:
    ys55_auc = None

print(f"Precision_ys55: {ys55_prec:.4f}")
print(f"Recall_ys55   : {ys55_rec:.4f}")
print(f"F1 Score_ys55 : {ys55_f1:.4f}")
print(f"AUC-ROC_ys55 : {ys55_auc:.4f}")

# ------------------------------------------------------------
# 5. ROC Curve
# ------------------------------------------------------------

if ys55_auc is not None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys55, tpr_ys55, label=f'AdaBoost (AUC = {ys55_auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve – AdaBoost (ys55)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------
# 6. Confusion Matrix Heatmap (Counts + Percentages)
# ------------------------------------------------------------

ys55_cm = confusion_matrix(y_test, ys55_pred)
ys55_cm_pct = ys55_cm.astype(float) / ys55_cm.sum(axis=1)[:, None] * 100

ys55_labels = np.array([
    [f"{ys55_cm[i,j]}\n({ys55_cm_pct[i,j]:.1f}%)" for j in range(ys55_cm.shape[1])]
    for i in range(ys55_cm.shape[0])
])

plt.figure(figsize=(6, 5))
sns.heatmap(ys55_cm, annot=ys55_labels, fmt="", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – AdaBoost (ys55)")
plt.show()

# ------------------------------------------------------------
# 7. Mean ± SD Across 5-Fold CV (Reviewer #2: Concern #19)
# ------------------------------------------------------------

ys55_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys55_cv = cross_validate(
    ys55_best_ab, X_train, y_train,
    cv=5, scoring=ys55_scoring
)

print("\n===== AdaBoost (ys55) – Mean ± SD over 5-Fold CV =====")
for metric in ys55_scoring:
    scores = ys55_cv[f"test_{metric}"]
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")
print("========================================================\n")

# ------------------------------------------------------------
# 8. Mean ± SD Visualization Plot
# ------------------------------------------------------------

ys55_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ys55_means = [
    ys55_cv['test_accuracy'].mean(),
    ys55_cv['test_precision'].mean(),
    ys55_cv['test_recall'].mean(),
    ys55_cv['test_f1'].mean(),
    ys55_cv['test_roc_auc'].mean()
]

ys55_stds = [
    ys55_cv['test_accuracy'].std(),
    ys55_cv['test_precision'].std(),
    ys55_cv['test_recall'].std(),
    ys55_cv['test_f1'].std(),
    ys55_cv['test_roc_auc'].std()
]

x = np.arange(len(ys55_metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys55_means,
    yerr=ys55_stds,
    fmt='o-', markersize=8, capsize=6, linewidth=2,
    color='purple', ecolor='violet'
)

plt.xticks(x, ys55_metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Mean ± SD – AdaBoost Classifier (ys55)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

```

### Cell 49 — markdown (execution_count=None)
```text
# **AdaBoost-SVM**
```

### Cell 50 — code (execution_count=None)
```python
'''# I AdaBoost-SVM
param_grid_absvm = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.1, 1.0, 1.1, 1.2]
}

# I AdaBoost SVM Grid Search
absvm = AdaBoostClassifier(SVC(probability=True)) # Changed: Added probability=True to SVC
grid_search_absvm = GridSearchCV(absvm, param_grid_absvm, cv=5, scoring='accuracy')
grid_search_absvm.fit(X_train, y_train)
best_absvm = grid_search_absvm.best_estimator_
# Output the best parameters
print("Best Parameters of AdaBoost SVM Grid Search:", best_absvm)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Fit the best model on the training data
best_absvm.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_absvm = best_absvm.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_absvm:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_absvm = cross_val_score(best_absvm, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_absvm = cv_scores_absvm.mean()
print(f"Validation Accuracy of Decision absvm: {validation_accuracy_absvm:.4f}")

# 3. Predictions on the test set (you need to have X_test and y_test)
y_pred_absvm = best_absvm.predict(X_test)

# 4. Calculate other peabsvmormance metrics
print("Confusion Matrix of absvm:")
print(confusion_matrix(y_test, y_pred_absvm))

print("Classification Report of absvm:")
print(classification_report(y_test, y_pred_absvm))

# Optional: If you want to calculate accuracy directly
test_accuracy_absvm = accuracy_score(y_test, y_pred_absvm)
print(f"Test Accuracy of absvm: {test_accuracy_absvm:.4f}")
'''
```

### Cell 51 — markdown (execution_count=None)
```text
# **XGBoost Classifier** *Reviewer 2 Concern 18*
```

### Cell 52 — code (execution_count=None)
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

### Cell 53 — code (execution_count=None)
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

### Cell 54 — code (execution_count=None)
```python
# ============================================================
# XGBoost Classifier (ys56)
# Full Pipeline: Grid Search → Evaluation → ROC → CM → Mean±SD → Plot
# Reviewer #2 Concern #19 Fully Addressed
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# 1. XGBoost Grid Search (ys56)
# ------------------------------------------------------------

ys56_param_grid = {
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

ys56_xgb = XGBClassifier()

ys56_grid_search = GridSearchCV(
    ys56_xgb, ys56_param_grid,
    cv=5, scoring='accuracy'
)

ys56_grid_search.fit(X_train, y_train)
ys56_best_xgb = ys56_grid_search.best_estimator_

print("\nBest XGBoost Parameters (ys56):")
print(ys56_best_xgb)

# ------------------------------------------------------------
# 2. Fit Best Model
# ------------------------------------------------------------

ys56_best_xgb.fit(X_train, y_train)

# Training accuracy
ys56_train_acc = ys56_best_xgb.score(X_train, y_train)
print(f"\nTraining Accuracy_ys56: {ys56_train_acc:.4f}")

# 5-fold CV Validation Accuracy
ys56_val_acc = cross_val_score(
    ys56_best_xgb, X_train, y_train,
    cv=5, scoring='accuracy'
).mean()
print(f"Validation Accuracy_ys56: {ys56_val_acc:.4f}")

# ------------------------------------------------------------
# 3. Predictions on Test Set
# ------------------------------------------------------------

ys56_pred = ys56_best_xgb.predict(X_test)

print("\nConfusion Matrix_ys56:")
print(confusion_matrix(y_test, ys56_pred))

print("\nClassification Report_ys56:")
print(classification_report(y_test, ys56_pred))

ys56_test_acc = accuracy_score(y_test, ys56_pred)
print(f"Testing Accuracy_ys56: {ys56_test_acc:.4f}")

# ------------------------------------------------------------
# 4. Precision, Recall, F1, AUC
# ------------------------------------------------------------

ys56_prec = precision_score(y_test, ys56_pred)
ys56_rec = recall_score(y_test, ys56_pred)
ys56_f1 = f1_score(y_test, ys56_pred)

if hasattr(ys56_best_xgb, "predict_proba"):
    ys56_prob = ys56_best_xgb.predict_proba(X_test)[:, 1]
    fpr_ys56, tpr_ys56, _ = roc_curve(y_test, ys56_prob)
    ys56_auc = auc(fpr_ys56, tpr_ys56)
else:
    ys56_auc = None

print(f"Precision_ys56: {ys56_prec:.4f}")
print(f"Recall_ys56   : {ys56_rec:.4f}")
print(f"F1 Score_ys56 : {ys56_f1:.4f}")
print(f"AUC-ROC_ys56 : {ys56_auc:.4f}")

# ------------------------------------------------------------
# 5. ROC Curve
# ------------------------------------------------------------

if ys56_auc is not None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys56, tpr_ys56, label=f'XGBoost (AUC = {ys56_auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve – XGBoost (ys56)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------
# 6. Confusion Matrix Heatmap (Counts + Percentages)
# ------------------------------------------------------------

ys56_cm = confusion_matrix(y_test, ys56_pred)
ys56_cm_pct = ys56_cm.astype(float) / ys56_cm.sum(axis=1)[:, None] * 100

ys56_labels = np.array([
    [f"{ys56_cm[i,j]}\n({ys56_cm_pct[i,j]:.1f}%)" for j in range(ys56_cm.shape[1])]
    for i in range(ys56_cm.shape[0])
])

plt.figure(figsize=(6, 5))
sns.heatmap(ys56_cm, annot=ys56_labels, fmt="", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – XGBoost (ys56)")
plt.show()

# ------------------------------------------------------------
# 7. Mean ± SD Across 5-Fold CV (Reviewer #2: Concern #19)
# ------------------------------------------------------------

ys56_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys56_cv = cross_validate(
    ys56_best_xgb, X_train, y_train,
    cv=5, scoring=ys56_scoring
)

print("\n===== XGBoost (ys56) – Mean ± SD over 5-Fold CV =====")
for metric in ys56_scoring:
    scores = ys56_cv[f"test_{metric}"]
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")
print("========================================================\n")

# ------------------------------------------------------------
# 8. Mean ± SD Visualization Plot
# ------------------------------------------------------------

ys56_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ys56_means = [
    ys56_cv['test_accuracy'].mean(),
    ys56_cv['test_precision'].mean(),
    ys56_cv['test_recall'].mean(),
    ys56_cv['test_f1'].mean(),
    ys56_cv['test_roc_auc'].mean()
]

ys56_stds = [
    ys56_cv['test_accuracy'].std(),
    ys56_cv['test_precision'].std(),
    ys56_cv['test_recall'].std(),
    ys56_cv['test_f1'].std(),
    ys56_cv['test_roc_auc'].std()
]

x = np.arange(len(ys56_metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys56_means,
    yerr=ys56_stds,
    fmt='o-', markersize=8, capsize=6, linewidth=2,
    color='darkorange', ecolor='gold'
)

plt.xticks(x, ys56_metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Mean ± SD – XGBoost Classifier (ys56)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

```

### Cell 55 — markdown (execution_count=None)
```text
# **XGBoost Regressor**
```

### Cell 56 — code (execution_count=None)
```python
'''
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Parameter grid for XGBoost Classifier
param_grid_xgb_reg = {
    'nthread': [4],
    'objective': ['binary:logistic'],  # Use binary classification objective
    'max_depth': [5, 6, 7],
    'min_child_weight': [4],
    'silent': [1],
    'subsample': [0.7],
    'colsample_bytree': [0.7]
}

# XGBoost Classifier Grid Search
xgb_reg = XGBClassifier()
grid_search_xgb_reg = GridSearchCV(xgb_reg, param_grid_xgb_reg, cv=5, scoring='accuracy')
grid_search_xgb_reg.fit(X_train, y_train)
best_xgb_reg = grid_search_xgb_reg.best_estimator_
print("Best Parameters of XGBoost Regressor Grid Search:", best_xgb_reg)

# Fit the best model on the training data
best_xgb_reg.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_xgb_reg = best_xgb_reg.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_xgb_reg:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_xgb_reg = cross_val_score(best_xgb_reg, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_xgb_reg = cv_scores_xgb_reg.mean()
print(f"Validation Accuracy: {validation_accuracy_xgb_reg:.4f}")

# 3. Predictions on the test set (regification)
y_pred_xgb_reg = best_xgb_reg.predict(X_test)

# 4. Calculate regification performance metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb_reg))

print("Classification Report:")
print(classification_report(y_test, y_pred_xgb_reg))

# Optional: Calculate test accuracy directly
test_accuracy_xgb_reg = accuracy_score(y_test, y_pred_xgb_reg)
print(f"Test Accuracy: {test_accuracy_xgb_reg:.4f}")
'''
```

### Cell 57 — markdown (execution_count=None)
```text
# F SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# F SVM Grid Search
svm = SVC()
grid_search_svc = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)
best_svc = grid_search_svc.best_estimator_
print("Best Parameters of SVM Grid Search:", best_svc)

```

### Cell 58 — markdown (execution_count=None)
```text
# **Lightgbm**, *Reviewer 2, Concern 18*
```

### Cell 59 — code (execution_count=None)
```python
import lightgbm as gbm
lgbm = gbm.LGBMClassifier()

# Fit the best model on the training data
lgbm.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_lgbm = lgbm.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_lgbm:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_lgbm = cross_val_score(lgbm, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_lgbm = cv_scores_lgbm.mean()
print(f"Validation Accuracy: {validation_accuracy_lgbm:.4f}")

# 3. Predictions on the test set (classification)
y_pred_lgbm = lgbm.predict(X_test)

# 4. Calculate classification performance metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lgbm))

print("Classification Report:")
print(classification_report(y_test, y_pred_lgbm))

# Optional: Calculate test accuracy directly
test_accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
print(f"Test Accuracy: {test_accuracy_lgbm:.4f}")
```

### Cell 60 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys56 Classifier
lgbm.fit(X_train, y_train)

# Make predictions
ys56_predictions = lgbm.predict(X_test)

# Calculate metrics
train_accuracy_ys56 = lgbm.score(X_train, y_train)
test_accuracy_ys56 = accuracy_score(y_test, ys56_predictions)
precision_ys56 = precision_score(y_test, ys56_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys56 = recall_score(y_test, ys56_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys56 = f1_score(y_test, ys56_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys56 would be needed for using probabilities)
if hasattr(lgbm, "predict_proba"):
    ys56_probabilities = lgbm.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys56, tpr_ys56, thresholds = roc_curve(y_test, ys56_probabilities)
    auc_score = auc(fpr_ys56, tpr_ys56)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys56, tpr_ys56, color='blue', label=f'Stacking Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys56."

# Display all metrics
print(f'Training Accuracy_ys56: {train_accuracy_ys56:.4f}')
print(f'Testing Accuracy_ys56: {test_accuracy_ys56:.4f}')
print(f'Precision_ys56: {precision_ys56:.4f}')
print(f'Recall_ys56: {recall_ys56:.4f}')
print(f'F1_ys56 Score: {f1_ys56:.4f}')
print(f'AUC-ROC_ys56: {auc_score}')


# Compute confusion matrix
cm_ys56 = confusion_matrix(y_test, ys56_predictions)

# Calculate percentages
cm_ys56_percent = cm_ys56.astype('float') / cm_ys56.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys56, cm_ys56_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys56, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 61 — code (execution_count=None)
```python
# ============================================================
# LightGBM Classifier (ys57)
# Full Pipeline: Fit → Metrics → ROC → CM → Mean±SD → Plot
# Reviewer #2 Concern #19 Fully Satisfied
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

import lightgbm as gbm

# ------------------------------------------------------------
# 1. Train the Model (ys57)
# ------------------------------------------------------------

ys57_lgbm = gbm.LGBMClassifier()
ys57_lgbm.fit(X_train, y_train)

# Training accuracy
ys57_train_acc = ys57_lgbm.score(X_train, y_train)
print(f"\nTraining Accuracy_ys57: {ys57_train_acc:.4f}")

# 5-fold validation accuracy
ys57_val_acc = cross_val_score(
    ys57_lgbm, X_train, y_train, cv=5, scoring='accuracy'
).mean()
print(f"Validation Accuracy_ys57: {ys57_val_acc:.4f}")

# ------------------------------------------------------------
# 2. Make Predictions
# ------------------------------------------------------------

ys57_pred = ys57_lgbm.predict(X_test)

print("\nConfusion Matrix_ys57:")
print(confusion_matrix(y_test, ys57_pred))

print("\nClassification Report_ys57:")
print(classification_report(y_test, ys57_pred))

ys57_test_acc = accuracy_score(y_test, ys57_pred)
print(f"Testing Accuracy_ys57: {ys57_test_acc:.4f}")

# ------------------------------------------------------------
# 3. Precision, Recall, F1, AUC
# ------------------------------------------------------------

ys57_prec = precision_score(y_test, ys57_pred)
ys57_rec = recall_score(y_test, ys57_pred)
ys57_f1 = f1_score(y_test, ys57_pred)

if hasattr(ys57_lgbm, "predict_proba"):
    ys57_prob = ys57_lgbm.predict_proba(X_test)[:, 1]
    fpr_ys57, tpr_ys57, _ = roc_curve(y_test, ys57_prob)
    ys57_auc = auc(fpr_ys57, tpr_ys57)
else:
    ys57_auc = None

print(f"Precision_ys57: {ys57_prec:.4f}")
print(f"Recall_ys57   : {ys57_rec:.4f}")
print(f"F1 Score_ys57 : {ys57_f1:.4f}")
print(f"AUC-ROC_ys57 : {ys57_auc:.4f}")

# ------------------------------------------------------------
# 4. ROC Curve Plot
# ------------------------------------------------------------

if ys57_auc is not None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys57, tpr_ys57, label=f'LightGBM (AUC = {ys57_auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve – LightGBM (ys57)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------
# 5. Confusion Matrix Heatmap (Counts + Percentages)
# ------------------------------------------------------------

ys57_cm = confusion_matrix(y_test, ys57_pred)
ys57_cm_pct = ys57_cm.astype(float) / ys57_cm.sum(axis=1)[:, None] * 100

ys57_labels = np.array([
    [f"{ys57_cm[i,j]}\n({ys57_cm_pct[i,j]:.1f}%)" for j in range(ys57_cm.shape[1])]
    for i in range(ys57_cm.shape[0])
])

plt.figure(figsize=(6, 5))
sns.heatmap(ys57_cm, annot=ys57_labels, fmt="", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – LightGBM (ys57)")
plt.show()

# ------------------------------------------------------------
# 6. Mean ± SD Across 5-Fold CV (Reviewer #2: Concern #19)
# ------------------------------------------------------------

ys57_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys57_cv = cross_validate(
    ys57_lgbm, X_train, y_train,
    cv=5, scoring=ys57_scoring
)

print("\n===== LightGBM (ys57) – Mean ± SD over 5-Fold CV =====")
for metric in ys57_scoring:
    scores = ys57_cv[f"test_{metric}"]
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")
print("========================================================\n")

# ------------------------------------------------------------
# 7. Mean ± SD Visualization Plot
# ------------------------------------------------------------

ys57_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ys57_means = [
    ys57_cv['test_accuracy'].mean(),
    ys57_cv['test_precision'].mean(),
    ys57_cv['test_recall'].mean(),
    ys57_cv['test_f1'].mean(),
    ys57_cv['test_roc_auc'].mean()
]

ys57_stds = [
    ys57_cv['test_accuracy'].std(),
    ys57_cv['test_precision'].std(),
    ys57_cv['test_recall'].std(),
    ys57_cv['test_f1'].std(),
    ys57_cv['test_roc_auc'].std()
]

x = np.arange(len(ys57_metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys57_means,
    yerr=ys57_stds,
    fmt='o-', markersize=8, capsize=6, linewidth=2,
    color='darkred', ecolor='lightcoral'
)

plt.xticks(x, ys57_metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Mean ± SD – LightGBM Classifier (ys57)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

```

### Cell 62 — markdown (execution_count=None)
```text
# SVM
```

### Cell 63 — code (execution_count=None)
```python
'''
svc=SVC(kernel='linear',C=1, probability=True)

# Fit the best model on the training data
svc.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_svc = svc.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_svc:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_svc = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_svc = cv_scores_svc.mean()
print(f"Validation Accuracy: {validation_accuracy_svc:.4f}")

# 3. Predictions on the test set (classification)
y_pred_svc = svc.predict(X_test)

# 4. Calculate classification performance metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svc))

print("Classification Report:")
print(classification_report(y_test, y_pred_svc))

# Optional: Calculate test accuracy directly
test_accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"Test Accuracy: {test_accuracy_svc:.4f}")
'''
```

### Cell 64 — markdown (execution_count=None)
```text
KNN
```

### Cell 65 — code (execution_count=None)
```python
'''
knn = KNeighborsClassifier(metric='manhattan', weights='distance')


# Fit the best model on the training data
knn.fit(X_train, y_train)

# 1. Calculate Training Accuracy
train_accuracy_knn = knn.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy_knn:.4f}")

# 2. Calculate Validation Accuracy (using cross-validation)
cv_scores_knn = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
validation_accuracy_knn = cv_scores_knn.mean()
print(f"Validation Accuracy: {validation_accuracy_knn:.4f}")

# 3. Predictions on the test set (classification)
y_pred_knn = knn.predict(X_test)

# 4. Calculate classification performance metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Optional: Calculate test accuracy directly
test_accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Test Accuracy: {test_accuracy_knn:.4f}")
'''
```

### Cell 66 — markdown (execution_count=None)
```text
# **CatBOOST**
```

### Cell 67 — code (execution_count=None)
```python
!pip install catboost

```

### Cell 68 — code (execution_count=None)
```python
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    log_loss, cohen_kappa_score, matthews_corrcoef,
    hamming_loss
)
from catboost import CatBoostClassifier

# ------------------------------
# Train CatBoost Classifier
# ------------------------------
CATmodel = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    verbose=False
)

start_time = time.time()
CATmodel.fit(X_train, y_train)
train_time = time.time() - start_time

# ------------------------------
# Predictions
# ------------------------------
y_pred = CATmodel.predict(X_test)
y_pred_proba = CATmodel.predict_proba(X_test)[:, 1]

# ------------------------------
# Metrics
# ------------------------------

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
}

# ------------------------------
# Display Output
# ------------------------------
print("\n📌 CatBoost Model Evaluation Metrics\n")
for k, v in metrics.items():
    print(f"{k:20s}: {v:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

```

### Cell 69 — code (execution_count=None)
```python
# ------------------------------
# Predictions
# ------------------------------
ys57_predictions = CATmodel.predict(X_test)
ys57_probabilities = CATmodel.predict_proba(X_test)[:, 1]  # Positive class probability

# ------------------------------
# Metrics
# ------------------------------
train_accuracy_ys57 = CATmodel.score(X_train, y_train)
test_accuracy_ys57 = accuracy_score(y_test, ys57_predictions)
precision_ys57 = precision_score(y_test, ys57_predictions)
recall_ys57 = recall_score(y_test, ys57_predictions)
f1_ys57 = f1_score(y_test, ys57_predictions)
auc_score_ys57 = roc_auc_score(y_test, ys57_probabilities)

# ------------------------------
# ROC Curve
# ------------------------------
fpr_ys57, tpr_ys57, _ = roc_curve(y_test, ys57_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr_ys57, tpr_ys57, label=f'CATBoost Classifier AUC = {auc_score_ys57:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--')  # Random line
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ------------------------------
# Display Metrics
# ------------------------------
print(f"Training Accuracy_ys57: {train_accuracy_ys57:.4f}")
print(f"Testing Accuracy_ys57: {test_accuracy_ys57:.4f}")
print(f"Precision_ys57: {precision_ys57:.4f}")
print(f"Recall_ys57: {recall_ys57:.4f}")
print(f"F1_ys57 Score: {f1_ys57:.4f}")
print(f"AUC-ROC_ys57: {auc_score_ys57:.4f}")

# ------------------------------
# Confusion Matrix
# ------------------------------
cm_ys57 = confusion_matrix(y_test, ys57_predictions)

# Percentages
cm_ys57_percent = cm_ys57.astype('float') / cm_ys57.sum(axis=1)[:, np.newaxis] * 100

# Labels for heatmap
labels_ys57 = np.array([
    [f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
    for row, percent_row in zip(cm_ys57, cm_ys57_percent)
])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_ys57,
    annot=labels_ys57,
    fmt="",
    cmap="Blues",
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("CATBoost – Confusion Matrix (Counts & Percentages)")
plt.show()

```

### Cell 70 — code (execution_count=None)
```python
# ============================================================
# CatBoost Classifier (ys58)
# Full Pipeline: Train → Metrics → ROC → CM → Mean±SD → Plot
# Reviewer #2 Concern #19 Fully Satisfied
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
)

from catboost import CatBoostClassifier

# ------------------------------------------------------------
# 1. Train CatBoost (ys58)
# ------------------------------------------------------------

ys58_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    verbose=False
)

ys58_model.fit(X_train, y_train)

# Training accuracy
ys58_train_acc = ys58_model.score(X_train, y_train)
print(f"\nTraining Accuracy_ys58: {ys58_train_acc:.4f}")

# 5-fold validation accuracy
ys58_val_acc = cross_val_score(
    ys58_model, X_train, y_train,
    cv=5, scoring='accuracy'
).mean()
print(f"Validation Accuracy_ys58: {ys58_val_acc:.4f}")

# ------------------------------------------------------------
# 2. Predictions on Test Set
# ------------------------------------------------------------

ys58_pred = ys58_model.predict(X_test)
ys58_prob = ys58_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix_ys58:")
print(confusion_matrix(y_test, ys58_pred))

print("\nClassification Report_ys58:")
print(classification_report(y_test, ys58_pred))

ys58_test_acc = accuracy_score(y_test, ys58_pred)
print(f"Testing Accuracy_ys58: {ys58_test_acc:.4f}")

# ------------------------------------------------------------
# 3. Precision, Recall, F1, AUC
# ------------------------------------------------------------

ys58_prec = precision_score(y_test, ys58_pred)
ys58_rec = recall_score(y_test, ys58_pred)
ys58_f1 = f1_score(y_test, ys58_pred)
ys58_auc = roc_auc_score(y_test, ys58_prob)

print(f"Precision_ys58: {ys58_prec:.4f}")
print(f"Recall_ys58   : {ys58_rec:.4f}")
print(f"F1 Score_ys58 : {ys58_f1:.4f}")
print(f"AUC-ROC_ys58 : {ys58_auc:.4f}")

# ------------------------------------------------------------
# 4. ROC Curve Plot
# ------------------------------------------------------------

fpr_ys58, tpr_ys58, _ = roc_curve(y_test, ys58_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_ys58, tpr_ys58, label=f'CatBoost (AUC = {ys58_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve – CatBoost (ys58)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 5. Confusion Matrix Heatmap (Counts + Percentages)
# ------------------------------------------------------------

ys58_cm = confusion_matrix(y_test, ys58_pred)
ys58_cm_pct = ys58_cm.astype(float) / ys58_cm.sum(axis=1)[:, None] * 100

ys58_labels = np.array([
    [f"{ys58_cm[i,j]}\n({ys58_cm_pct[i,j]:.1f}%)" for j in range(ys58_cm.shape[1])]
    for i in range(ys58_cm.shape[0])
])

plt.figure(figsize=(6, 5))
sns.heatmap(
    ys58_cm, annot=ys58_labels, fmt="", cmap="Blues",
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – CatBoost (ys58)")
plt.show()

# ------------------------------------------------------------
# 6. Mean ± SD Across 5-Fold CV (Reviewer #2: Concern #19)
# ------------------------------------------------------------

ys58_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys58_cv = cross_validate(
    ys58_model, X_train, y_train,
    cv=5, scoring=ys58_scoring
)

print("\n===== CatBoost (ys58) – Mean ± SD over 5-Fold CV =====")
for metric in ys58_scoring:
    scores = ys58_cv[f"test_{metric}"]
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")
print("========================================================\n")

# ------------------------------------------------------------
# 7. Mean ± SD Visualization Plot
# ------------------------------------------------------------

ys58_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ys58_means = [
    ys58_cv['test_accuracy'].mean(),
    ys58_cv['test_precision'].mean(),
    ys58_cv['test_recall'].mean(),
    ys58_cv['test_f1'].mean(),
    ys58_cv['test_roc_auc'].mean()
]

ys58_stds = [
    ys58_cv['test_accuracy'].std(),
    ys58_cv['test_precision'].std(),
    ys58_cv['test_recall'].std(),
    ys58_cv['test_f1'].std(),
    ys58_cv['test_roc_auc'].std()
]

x = np.arange(len(ys58_metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(
    x, ys58_means,
    yerr=ys58_stds,
    fmt='o-', markersize=8, capsize=6, linewidth=2,
    color='darkgreen', ecolor='lightgreen'
)

plt.xticks(x, ys58_metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Mean ± SD – CatBoost Classifier (ys58)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

```

### Cell 71 — markdown (execution_count=None)
```text
# **ML Individual Comparison**
```

### Cell 72 — code (execution_count=None)
```python
import seaborn as sns
compar=pd.DataFrame(columns=["algorithm","Accuracy"])
compar.loc[len(compar)] = ['LogisticRegression', test_accuracy_logistic]
#compar.loc[len(compar)] = ['KNeighborsClassifier', test_accuracy_knn]
compar.loc[len(compar)] = ['DecisionTreeClassifier', test_accuracy]
#compar.loc[len(compar)] = ['RandomForestClassifier', test_accuracy_rf]
#compar.loc[len(compar)] = ['svc', test_accuracy_svc]
compar.loc[len(compar)] = ['lightgbm', test_accuracy_lgbm]
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

### Cell 73 — markdown (execution_count=None)
```text
# **New Stacking for 04 base** *Reviewer 2, Concern 22*
```

### Cell 74 — code (execution_count=None)
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

### Cell 75 — code (execution_count=None)
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

### Cell 76 — code (execution_count=None)
```python
# --- Compute confusion matrix ---
cm_ys66 = confusion_matrix(y_test, ys66_predictions)

# Calculate percentages
cm_ys66_percent = cm_ys66.astype('float') / cm_ys66.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys66, cm_ys66_percent)])

# --- Plot confusion matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys66, annot=labels, fmt="", cmap="Blues", cbar=True,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
#plt.title("Confusion Matrix (Counts & Percentages)")

# --- Save as 300 DPI PDF ---
output_path = "/content/confusion_matrix_ys66.pdf"
plt.savefig(output_path, dpi=300, format='pdf', bbox_inches='tight')

plt.show()

# --- Download the file automatically ---
from google.colab import files
files.download(output_path)

```

### Cell 77 — code (execution_count=None)
```python
import seaborn as sns
compar=pd.DataFrame(columns=["algorithm","Accuracy"])
compar.loc[len(compar)] = ['LogisticRegression', test_accuracy_logistic]
compar.loc[len(compar)] = ['DecisionTreeClassifier', test_accuracy]
compar.loc[len(compar)] = ['lightgbm', test_accuracy_lgbm]
compar.loc[len(compar)] = ['Bayesian Networks', test_accuracy_bayes]
compar.loc[len(compar)] = ['AdaBoost', test_accuracy_ab]
compar.loc[len(compar)] = ['XGBoost Classifier', test_accuracy_xgb]
compar.loc[len(compar)] = ['Proposed_Stacking', test_accuracy_ys66]
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

### Cell 78 — code (execution_count=None)
```python
# ============================================================
# NOSEC Stacking Classifier (ys66)
# Full Pipeline: Train → Metrics → ROC → CM → Mean±SD → Plot
# Reviewer #2 Concern #19 Fully Satisfied
# ============================================================

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

# ------------------------------------------------------------
# 1. Build NOSEC Stacking Classifier (ys66)
# ------------------------------------------------------------

ys66_stacker = StackingClassifier(
    cv=5,
    estimators=[
        ('bayes', best_bayes),
        ('adaboost', best_ab),
        ('decision_tree', best_tree),
        ('logistic', best_logistic)
    ],
    final_estimator=best_xgb
)

# Calibrate stacker
ys66_model = CalibratedClassifierCV(ys66_stacker, method='sigmoid', cv=5)
ys66_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 2. Predictions
# ------------------------------------------------------------

ys66_pred = ys66_model.predict(X_test)
ys66_prob = ys66_model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix_ys66:")
print(confusion_matrix(y_test, ys66_pred))

print("\nClassification Report_ys66:")
print(classification_report(y_test, ys66_pred))

# ------------------------------------------------------------
# 3. Metrics
# ------------------------------------------------------------

ys66_train_acc = ys66_model.score(X_train, y_train)
ys66_test_acc = accuracy_score(y_test, ys66_pred)
ys66_prec = precision_score(y_test, ys66_pred)
ys66_rec = recall_score(y_test, ys66_pred)
ys66_f1 = f1_score(y_test, ys66_pred)
ys66_auc = auc(*roc_curve(y_test, ys66_prob)[:2])

print(f"Training Accuracy_ys66: {ys66_train_acc:.4f}")
print(f"Testing Accuracy_ys66 : {ys66_test_acc:.4f}")
print(f"Precision_ys66       : {ys66_prec:.4f}")
print(f"Recall_ys66          : {ys66_rec:.4f}")
print(f"F1 Score_ys66        : {ys66_f1:.4f}")
print(f"AUC-ROC_ys66         : {ys66_auc:.4f}")

# ------------------------------------------------------------
# 4. ROC Curve
# ------------------------------------------------------------

fpr_ys66, tpr_ys66, _ = roc_curve(y_test, ys66_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_ys66, tpr_ys66, color='blue', linewidth=2,
         label=f'NOSEC (AUC = {ys66_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve – NOSEC Stacking Classifier (ys66)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 5. Confusion Matrix Heatmap
# ------------------------------------------------------------

ys66_cm = confusion_matrix(y_test, ys66_pred)
ys66_pct = ys66_cm.astype(float) / ys66_cm.sum(axis=1)[:, None] * 100

ys66_labels = np.array([
    [f"{ys66_cm[i,j]}\n({ys66_pct[i,j]:.1f}%)"
     for j in range(ys66_cm.shape[1])]
    for i in range(ys66_cm.shape[0])
])

plt.figure(figsize=(6, 5))
sns.heatmap(
    ys66_cm, annot=ys66_labels, fmt="", cmap="Blues",
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)
plt.title("NOSEC – Confusion Matrix (ys66)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ------------------------------------------------------------
# 6. Mean ± SD (Reviewer #2 Requirement)
# ------------------------------------------------------------

ys66_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys66_cv = cross_validate(
    ys66_model, X_train, y_train,
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

### Cell 79 — code (execution_count=None)
```python

```

### Cell 80 — code (execution_count=None)
```python

```

### Cell 81 — markdown (execution_count=None)
```text
# **Confidence Intervals** *Reviewer 2, Concern 19*
```

### Cell 82 — code (execution_count=None)
```python
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function to compute metrics
def get_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

# Original predictions
y_pred = calibrated_stacking_xgb.predict(X_test)
y_proba = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]

# Get original metrics
original_metrics = get_metrics(y_test, y_pred, y_proba)

# Bootstrapping
n_bootstrap = 1000
rng = np.random.RandomState(42)
metrics_bootstrap = {m: [] for m in original_metrics.keys()}

for _ in range(n_bootstrap):
    # Resample test set
    indices = rng.randint(0, len(X_test), len(X_test))
    X_resampled = X_test[indices]
    y_resampled = y_test[indices]

    y_pred_resampled = calibrated_stacking_xgb.predict(X_resampled)
    y_proba_resampled = calibrated_stacking_xgb.predict_proba(X_resampled)[:, 1]

    m = get_metrics(y_resampled, y_pred_resampled, y_proba_resampled)
    for key in metrics_bootstrap.keys():
        metrics_bootstrap[key].append(m[key])

# Compute 95% CIs
ci_results = {}
for key in original_metrics.keys():
    lower = np.percentile(metrics_bootstrap[key], 2.5)
    upper = np.percentile(metrics_bootstrap[key], 97.5)
    ci_results[key] = (original_metrics[key], lower, upper)

# Display results
print("\nMetrics with 95% Confidence Intervals (Bootstrap 1000):")
for metric, (mean_val, ci_low, ci_high) in ci_results.items():
    print(f"{metric.capitalize():<10}: {mean_val:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")

```

### Cell 83 — markdown (execution_count=None)
```text
# Calibration (slope, intercept, Brier) and PR-AUC :::: Reviewer#2, Concern # 13:
```

### Cell 84 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression

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

# -------------------------------------------------------
# 1. Model predictions
# -------------------------------------------------------
y_probs = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]
y_pred = calibrated_stacking_xgb.predict(X_test)

# -------------------------------------------------------
# 2. Calibration Curve
# -------------------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', label="Calibrated Stacking Classifier")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve (Reliability Plot)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# 3. Brier Score
# -------------------------------------------------------
brier = brier_score_loss(y_test, y_probs)
print("Brier Score:", round(brier, 4))

# -------------------------------------------------------
# 4. Calibration Slope & Intercept
# -------------------------------------------------------
# According to standard calibration methods:
# Fit logistic regression: outcome ~ predicted log-odds

eps = 1e-7
log_odds = np.log(y_probs / (1 - y_probs + eps))

cal_model = LogisticRegression().fit(log_odds.reshape(-1,1), y_test)

cal_intercept = cal_model.intercept_[0]
cal_slope = cal_model.coef_[0][0]

print("Calibration Intercept:", round(cal_intercept, 4))
print("Calibration Slope:", round(cal_slope, 4))

# -------------------------------------------------------
# 5. PR-AUC
# -------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

print("PR-AUC:", round(pr_auc, 4))

# -------------------------------------------------------
# 6. Decision Curve Analysis (DCA)
# -------------------------------------------------------
def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y_true)
    net_benefit = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp/n) - (fp/n) * (t/(1-t))
        net_benefit.append(nb)

    return thresholds, net_benefit

thresholds, net_benefit = decision_curve_analysis(y_test, y_probs)

# Print threshold → net benefit mapping
print("\nThreshold   Net Benefit")
print("-------------------------")
for t, nb in zip(thresholds, net_benefit):
    print(f"{t:.3f}        {nb:.6f}")


plt.figure(figsize=(6,5))
plt.plot(thresholds, net_benefit, label="Calibrated Stacking Classifier")
plt.plot(thresholds, [0]*len(thresholds), linestyle="--", label="Treat None")
plt.plot(thresholds, [np.mean(y_test)]*len(thresholds), linestyle="--", label="Treat All")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# 5b. Precision-Recall Curve
# -------------------------------------------------------
plt.figure(figsize=(6,5))
plt.plot(recall, precision, color='b', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

```

### Cell 85 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from google.colab import files

# -------------------------------------------------------
# Function to save and download a plot
# -------------------------------------------------------
def save_and_download_plot(filename):
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    files.download(filename)  # triggers download in Colab

# -------------------------------------------------------
# 2. Calibration Curve
# -------------------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', label="Platt Calibrated Proposed OSEC")
#plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
#plt.title("Calibration Curve (Reliability Plot)")
plt.legend()
plt.legend(loc='upper left')
plt.grid(True)
save_and_download_plot("calibration_curve.pdf")
plt.close()

# -------------------------------------------------------
# 5b. Precision-Recall Curve
# -------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6,5))
plt.plot(recall, precision, color='b', lw=1.8, label=f'Platt Calibrated Proposed OSEC PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
#plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
save_and_download_plot("precision_recall_curve.pdf")
plt.close()

# -------------------------------------------------------
# 6. Decision Curve Analysis (DCA)
# -------------------------------------------------------
def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y_true)
    net_benefit = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp/n) - (fp/n) * (t/(1-t))
        net_benefit.append(nb)

    return thresholds, net_benefit

thresholds, net_benefit = decision_curve_analysis(y_test, y_probs)
plt.figure(figsize=(6,5))
plt.plot(thresholds, net_benefit, label="Proposed Optimized Stacking Ensemble Classifier")
plt.plot(thresholds, [0]*len(thresholds), linestyle="--", label="Treat None")
plt.plot(thresholds, [np.mean(y_test)]*len(thresholds), linestyle="--", label="Treat All")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
#plt.title("Decision Curve Analysis (DCA)")
plt.legend()
plt.legend(loc='lower left')
plt.grid(True)
save_and_download_plot("decision_curve.pdf")
plt.close()

```

### Cell 86 — code (execution_count=None)
```python
# ------------------------------------------------------------
# 3. Compute DCA Values
# ------------------------------------------------------------
print(f'Test labels shape: {y_test_old.shape}')
# True labels
y_true = y_test_old

thresholds, net_benefit_nosec = decision_curve_analysis(y_true, y_prob)

# ------------------------------------------------------------
# 3A. Print Net Benefit Along With Threshold
# ------------------------------------------------------------
print("\n=== Threshold vs Net Benefit (OSEC Voting Classifier) ===")
print(f"{'Threshold':<12} {'Net Benefit':<12}")

for t, nb in zip(thresholds, net_benefit_nosec):
    print(f"{t:<12.4f} {nb:<12.6f}")

# Treat None strategy (always zero)
nb_treat_none = np.zeros_like(thresholds)

# Treat All strategy (depends on prevalence)
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

```

### Cell 87 — code (execution_count=None)
```python

```

### Cell 88 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# -----------------------------
# 1. Get model predictions (Calibrated Stacking Classifier)
# -----------------------------
y_probs = calibrated_stacking_xgb.predict_proba(X_test)[:, 1]  # Probabilities for positive class
y_pred = calibrated_stacking_xgb.predict(X_test)

# -----------------------------
# 2. Calibration Curve
# -----------------------------
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', label="Calibrated Stacking Classifier")
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
print("Brier Score (Calibrated Stacking):", round(brier, 4))

# -----------------------------
# 4. Decision Curve Analysis (DCA)
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
plt.plot(thresholds, net_benefit, label="Calibrated Stacking Classifier")
plt.plot(thresholds, [0]*len(thresholds), linestyle="--", label="Treat None")
plt.plot(thresholds, [np.mean(y_test)]*len(thresholds), linestyle="--", label="Treat All")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (DCA)")
plt.legend()
plt.grid(True)
plt.show()

```

### Cell 89 — code (execution_count=None)
```python

```

### Cell 90 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

# -----------------------------
# Settings
# -----------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)

# -----------------------------
# Helper: metric wrapper to avoid zero-division issues
# -----------------------------
def safe_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def safe_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

# -----------------------------
# 1. Bootstrap metric functions that call the model on resampled indices
# -----------------------------
def bootstrap_ci_model_metric(model, X, y, metric_func, n_bootstrap=1000, ci=95):
    """
    Compute metric (point estimate + bootstrap CI) by resampling indices and predicting
    using `model` each iteration. metric_func should accept (y_true, y_pred) or (y_true, y_prob)
    depending on metric; ROC-AUC expects probabilities here.
    """
    n = len(y)
    scores = []
    rng = np.random.RandomState(RNG_SEED)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)  # sample indices with replacement
        X_res = X[idx]
        y_res = y[idx]

        # get predictions/probabilities from model for the resampled X
        y_pred_res = model.predict(X_res)
        if hasattr(model, "predict_proba"):
            y_prob_res = model.predict_proba(X_res)[:, 1]
        else:
            # fallback: use decision_function if available, else use y_pred_res
            if hasattr(model, "decision_function"):
                try:
                    scores_df = model.decision_function(X_res)
                    # If binary, decision_function returns shape (n,) or (n,1)
                    y_prob_res = scores_df
                except Exception:
                    y_prob_res = y_pred_res
            else:
                y_prob_res = y_pred_res

        # Choose metric evaluation type
        if metric_func == roc_auc_score:
            # if all y_res are single class, skip and append nan
            try:
                s = metric_func(y_res, y_prob_res)
            except Exception:
                s = np.nan
        else:
            # metric expects (y_true, y_pred)
            s = metric_func(y_res, y_pred_res)

        scores.append(s)

    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]  # drop any nan (possible for ROC-AUC when resample has single class)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), (lower, upper)

def bootstrap_brier_model(model, X, y, n_bootstrap=1000, ci=95):
    n = len(y)
    scores = []
    rng = np.random.RandomState(RNG_SEED)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        X_res = X[idx]
        y_res = y[idx]

        if hasattr(model, "predict_proba"):
            y_prob_res = model.predict_proba(X_res)[:, 1]
        else:
            # fallback: use decision_function or predictions
            if hasattr(model, "decision_function"):
                try:
                    y_prob_res = model.decision_function(X_res)
                except Exception:
                    y_prob_res = model.predict(X_res)
            else:
                y_prob_res = model.predict(X_res)

        scores.append(brier_score_loss(y_res, y_prob_res))

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), (lower, upper)

# -----------------------------
# 2. Decision Curve Analysis + bootstrap for CI
# -----------------------------
def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y_true)
    net_benefit = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(nb)
    return np.array(thresholds), np.array(net_benefit)

def bootstrap_dca_model(model, X, y, n_bootstrap=500, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y)
    rng = np.random.RandomState(RNG_SEED)
    all_nb = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        X_res = X[idx]
        y_res = y[idx]
        if hasattr(model, "predict_proba"):
            y_prob_res = model.predict_proba(X_res)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob_res = model.decision_function(X_res)
            else:
                y_prob_res = model.predict(X_res)

        _, nb = decision_curve_analysis(y_res, y_prob_res, thresholds)
        all_nb.append(nb)

    all_nb = np.array(all_nb)
    mean_nb = np.mean(all_nb, axis=0)
    lower = np.percentile(all_nb, 2.5, axis=0)
    upper = np.percentile(all_nb, 97.5, axis=0)
    return mean_nb, lower, upper

# -----------------------------
# 3. Models to evaluate (updated set)
# -----------------------------
models = {
    "Logistic Regression": best_logistic,          # trained best logistic model
    "Decision Tree": best_tree,                    # trained best decision tree
    "Bayesian Networks": best_bayes,               # trained bayes model
    "AdaBoost": best_ab,                           # trained AdaBoost
    "XGBoost Classifier": best_xgb,                # trained XGBoost
    "Proposed Stacking": calibrated_stacking_xgb    # your final stacked calibrated model
}

# -----------------------------
# 4. Evaluate each model and collect bootstrap CIs
# -----------------------------
results = []

for name, model in models.items():
    print(f"Evaluating: {name}")

    # Predictions
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test)
        else:
            y_probs = y_pred

    # Point estimates
    acc_pt = accuracy_score(y_test, y_pred)
    prec_pt = precision_score(y_test, y_pred, zero_division=0)
    rec_pt = recall_score(y_test, y_pred, zero_division=0)
    f1_pt = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc_pt = roc_auc_score(y_test, y_probs)
    except Exception:
        auc_pt = np.nan
    brier_pt = brier_score_loss(y_test, y_probs)

    # Bootstrap CI
    acc_mean, acc_ci = bootstrap_ci_model_metric(model, X_test, y_test, accuracy_score, n_bootstrap=1000)
    prec_mean, prec_ci = bootstrap_ci_model_metric(model, X_test, y_test, safe_precision, n_bootstrap=1000)
    rec_mean, rec_ci = bootstrap_ci_model_metric(model, X_test, y_test, safe_recall, n_bootstrap=1000)
    f1_mean, f1_ci = bootstrap_ci_model_metric(model, X_test, y_test, safe_f1, n_bootstrap=1000)
    auc_mean, auc_ci = bootstrap_ci_model_metric(model, X_test, y_test, roc_auc_score, n_bootstrap=1000)
    brier_mean, brier_ci = bootstrap_brier_model(model, X_test, y_test, n_bootstrap=1000)

    results.append([
        name,
        f"{acc_pt:.3f} ({acc_ci[0]:.3f}–{acc_ci[1]:.3f})",
        f"{prec_pt:.3f} ({prec_ci[0]:.3f}–{prec_ci[1]:.3f})",
        f"{rec_pt:.3f} ({rec_ci[0]:.3f}–{rec_ci[1]:.3f})",
        f"{f1_pt:.3f} ({f1_ci[0]:.3f}–{f1_ci[1]:.3f})",
        f"{(auc_pt if not np.isnan(auc_pt) else np.nan):.3f} ({auc_ci[0]:.3f}–{auc_ci[1]:.3f})" if not np.isnan(auc_pt) else "nan",
        f"{brier_pt:.3f} ({brier_ci[0]:.3f}–{brier_ci[1]:.3f})"
    ])

# -----------------------------
# 5. Summary table
# -----------------------------
df_results = pd.DataFrame(results, columns=[
    "Model", "Accuracy (pt & 95% CI)", "Precision (pt & 95% CI)",
    "Recall (pt & 95% CI)", "F1 (pt & 95% CI)", "ROC-AUC (pt & 95% CI)",
    "Brier (pt & 95% CI)"
])
print("\n=== Model Performance with 95% Bootstrap CIs ===")
print(df_results.to_string(index=False))

# -----------------------------
# 6. Calibration curves (all models)
# -----------------------------
plt.figure(figsize=(7,6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test)
        else:
            y_probs = model.predict(X_test)
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.plot([0,1],[0,1], linestyle='--', color='black', label='Perfect Calibration')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curves")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# -----------------------------
# 7. Decision Curve Analysis with 95% CI shading
# -----------------------------
plt.figure(figsize=(8,6))
thresholds = np.linspace(0.01, 0.99, 50)

for name, model in models.items():
    mean_nb, lower_nb, upper_nb = bootstrap_dca_model(model, X_test, y_test, n_bootstrap=500, thresholds=thresholds)
    plt.plot(thresholds, mean_nb, label=name)
    plt.fill_between(thresholds, lower_nb, upper_nb, alpha=0.15)

plt.plot(thresholds, [0]*len(thresholds), linestyle="--", color="black", label="Treat None")
plt.plot(thresholds, [np.mean(y_test)]*len(thresholds), linestyle="--", color="gray", label="Treat All")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (95% bootstrap CI)")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

```

### Cell 91 — code (execution_count=None)
```python
# Bootstrap metric (n=700) functions that call the model on resampled indices
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

# -----------------------------
# Settings
# -----------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)

# -----------------------------
# Helper: metric wrapper to avoid zero-division issues
# -----------------------------
def safe_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)

def safe_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)

def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

# -----------------------------
# 1. Bootstrap metric functions that call the model on resampled indices
# -----------------------------
def bootstrap_ci_model_metric(model, X, y, metric_func, n_bootstrap=700, ci=95):
    """
    Compute metric (point estimate + bootstrap CI) by resampling indices and predicting
    using `model` each iteration. metric_func should accept (y_true, y_pred) or (y_true, y_prob)
    depending on metric; ROC-AUC expects probabilities here.
    """
    n = len(y)
    scores = []
    rng = np.random.RandomState(RNG_SEED)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)  # sample indices with replacement
        X_res = X[idx]
        y_res = y[idx]

        # get predictions/probabilities from model for the resampled X
        y_pred_res = model.predict(X_res)
        if hasattr(model, "predict_proba"):
            y_prob_res = model.predict_proba(X_res)[:, 1]
        else:
            # fallback: use decision_function if available, else use y_pred_res
            if hasattr(model, "decision_function"):
                try:
                    scores_df = model.decision_function(X_res)
                    # If binary, decision_function returns shape (n,) or (n,1)
                    y_prob_res = scores_df
                except Exception:
                    y_prob_res = y_pred_res
            else:
                y_prob_res = y_pred_res

        # Choose metric evaluation type
        if metric_func == roc_auc_score:
            # if all y_res are single class, skip and append nan
            try:
                s = metric_func(y_res, y_prob_res)
            except Exception:
                s = np.nan
        else:
            # metric expects (y_true, y_pred)
            s = metric_func(y_res, y_pred_res)

        scores.append(s)

    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]  # drop any nan (possible for ROC-AUC when resample has single class)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), (lower, upper)

def bootstrap_brier_model(model, X, y, n_bootstrap=1000, ci=95):
    n = len(y)
    scores = []
    rng = np.random.RandomState(RNG_SEED)

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        X_res = X[idx]
        y_res = y[idx]

        if hasattr(model, "predict_proba"):
            y_prob_res = model.predict_proba(X_res)[:, 1]
        else:
            # fallback: use decision_function or predictions
            if hasattr(model, "decision_function"):
                try:
                    y_prob_res = model.decision_function(X_res)
                except Exception:
                    y_prob_res = model.predict(X_res)
            else:
                y_prob_res = model.predict(X_res)

        scores.append(brier_score_loss(y_res, y_prob_res))

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), (lower, upper)

# -----------------------------
# 2. Decision Curve Analysis + bootstrap for CI
# -----------------------------
def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y_true)
    net_benefit = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        net_benefit.append(nb)
    return np.array(thresholds), np.array(net_benefit)

def bootstrap_dca_model(model, X, y, n_bootstrap=500, thresholds=np.linspace(0.01, 0.99, 50)):
    n = len(y)
    rng = np.random.RandomState(RNG_SEED)
    all_nb = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        X_res = X[idx]
        y_res = y[idx]
        if hasattr(model, "predict_proba"):
            y_prob_res = model.predict_proba(X_res)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob_res = model.decision_function(X_res)
            else:
                y_prob_res = model.predict(X_res)

        _, nb = decision_curve_analysis(y_res, y_prob_res, thresholds)
        all_nb.append(nb)

    all_nb = np.array(all_nb)
    mean_nb = np.mean(all_nb, axis=0)
    lower = np.percentile(all_nb, 2.5, axis=0)
    upper = np.percentile(all_nb, 97.5, axis=0)
    return mean_nb, lower, upper

# -----------------------------
# 3. Models to evaluate (updated set)
# -----------------------------
models = {
    "Logistic Regression": best_logistic,          # trained best logistic model
    "Decision Tree": best_tree,                    # trained best decision tree
    "Bayesian Networks": best_bayes,               # trained bayes model
    "AdaBoost": best_ab,                           # trained AdaBoost
    "XGBoost Classifier": best_xgb,                # trained XGBoost
    "Proposed Stacking": calibrated_stacking_xgb    # your final stacked calibrated model
}

# -----------------------------
# 4. Evaluate each model and collect bootstrap CIs
# -----------------------------
results = []

for name, model in models.items():
    print(f"Evaluating: {name}")

    # Predictions
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test)
        else:
            y_probs = y_pred

    # Point estimates
    acc_pt = accuracy_score(y_test, y_pred)
    prec_pt = precision_score(y_test, y_pred, zero_division=0)
    rec_pt = recall_score(y_test, y_pred, zero_division=0)
    f1_pt = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc_pt = roc_auc_score(y_test, y_probs)
    except Exception:
        auc_pt = np.nan
    brier_pt = brier_score_loss(y_test, y_probs)

    # Bootstrap CI
    acc_mean, acc_ci = bootstrap_ci_model_metric(model, X_test, y_test, accuracy_score, n_bootstrap=1000)
    prec_mean, prec_ci = bootstrap_ci_model_metric(model, X_test, y_test, safe_precision, n_bootstrap=1000)
    rec_mean, rec_ci = bootstrap_ci_model_metric(model, X_test, y_test, safe_recall, n_bootstrap=1000)
    f1_mean, f1_ci = bootstrap_ci_model_metric(model, X_test, y_test, safe_f1, n_bootstrap=1000)
    auc_mean, auc_ci = bootstrap_ci_model_metric(model, X_test, y_test, roc_auc_score, n_bootstrap=1000)
    brier_mean, brier_ci = bootstrap_brier_model(model, X_test, y_test, n_bootstrap=1000)

    results.append([
        name,
        f"{acc_pt:.3f} ({acc_ci[0]:.3f}–{acc_ci[1]:.3f})",
        f"{prec_pt:.3f} ({prec_ci[0]:.3f}–{prec_ci[1]:.3f})",
        f"{rec_pt:.3f} ({rec_ci[0]:.3f}–{rec_ci[1]:.3f})",
        f"{f1_pt:.3f} ({f1_ci[0]:.3f}–{f1_ci[1]:.3f})",
        f"{(auc_pt if not np.isnan(auc_pt) else np.nan):.3f} ({auc_ci[0]:.3f}–{auc_ci[1]:.3f})" if not np.isnan(auc_pt) else "nan",
        f"{brier_pt:.3f} ({brier_ci[0]:.3f}–{brier_ci[1]:.3f})"
    ])

# -----------------------------
# 5. Summary table
# -----------------------------
df_results = pd.DataFrame(results, columns=[
    "Model", "Accuracy (pt & 95% CI)", "Precision (pt & 95% CI)",
    "Recall (pt & 95% CI)", "F1 (pt & 95% CI)", "ROC-AUC (pt & 95% CI)",
    "Brier (pt & 95% CI)"
])
print("\n=== Model Performance with 95% Bootstrap CIs ===")
print(df_results.to_string(index=False))

# -----------------------------
# 6. Calibration curves (all models)
# -----------------------------
plt.figure(figsize=(7,6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test)
        else:
            y_probs = model.predict(X_test)
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.plot([0,1],[0,1], linestyle='--', color='black', label='Perfect Calibration')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curves")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# -----------------------------
# 7. Decision Curve Analysis with 95% CI shading
# -----------------------------
plt.figure(figsize=(8,6))
thresholds = np.linspace(0.01, 0.99, 50)

for name, model in models.items():
    mean_nb, lower_nb, upper_nb = bootstrap_dca_model(model, X_test, y_test, n_bootstrap=500, thresholds=thresholds)
    plt.plot(thresholds, mean_nb, label=name)
    plt.fill_between(thresholds, lower_nb, upper_nb, alpha=0.15)

plt.plot(thresholds, [0]*len(thresholds), linestyle="--", color="black", label="Treat None")
plt.plot(thresholds, [np.mean(y_test)]*len(thresholds), linestyle="--", color="gray", label="Treat All")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis (95% bootstrap CI)")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

```

### Cell 92 — markdown (execution_count=None)
```text
# **New Voting for 04 base**
```

### Cell 93 — code (execution_count=None)
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

### Cell 94 — code (execution_count=None)
```python
data_old = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data_old
data_old.head()
```

### Cell 95 — code (execution_count=None)
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

### Cell 96 — code (execution_count=None)
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

### Cell 97 — code (execution_count=None)
```python
X_old = data_old.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y_old = data_old['HeartDisease']
print(data_old)
print("X",X_old)
print("y",y_old)

#Target Variables Values
data_old.HeartDisease.value_counts()
```

### Cell 98 — code (execution_count=None)
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

### Cell 99 — code (execution_count=None)
```python
X_old = data_old_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y_old = data_old_no_outliers['HeartDisease']


#Target Variables Values
data_old_no_outliers.HeartDisease.value_counts()
X_old.shape
```

### Cell 100 — code (execution_count=None)
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

### Cell 101 — code (execution_count=None)
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

### Cell 102 — code (execution_count=None)
```python
df_imputed_old_X = df_imputed_old.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
df_imputed_old_y = df_imputed_old['HeartDisease']
print(df_imputed_old_X.shape)
```

### Cell 103 — code (execution_count=None)
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

### Cell 104 — code (execution_count=None)
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

### Cell 105 — code (execution_count=None)
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

### Cell 106 — code (execution_count=None)
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
    'n_estimators': [2, None],
    'criterion': ['gini', 'entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train_old, y_train_old)
rf_best_rs = rf_rs.best_estimator_

# 3. KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train_old, y_train_old)
knn_best_rs = knn_rs.best_estimator_

# 4. SVM
param_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train_old, y_train_old)
svm_best_rs = svm_rs.best_estimator_

```

### Cell 107 — code (execution_count=None)
```python
# Print Best Parameters
print("Logistic Regression Best Parameters:", logistic_best_rs)
print("Random Forest Best Parameters:", rf_best_rs)
print("K-Nearest Neighbors Best Parameters:", knn_best_rs)
print("Support Vector Machine Best Parameters:", svm_best_rs)

```

### Cell 108 — markdown (execution_count=None)
```text

```

### Cell 109 — code (execution_count=None)
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

### Cell 110 — code (execution_count=None)
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

### Cell 111 — code (execution_count=None)
```python
from sklearn.metrics import confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test_old, voting_predictions)
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

### Cell 112 — code (execution_count=None)
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files  # Needed for Colab download

# Example confusion matrix
cm_novec = np.array([[117, 14], [13, 131]])

# Maximum value for color scaling
max_val = cm_novec.max()

def plot_cm_save_download(cm, title, filename):
    # Calculate row-wise percentages
    total = np.sum(cm, axis=1)[:, np.newaxis]
    percentages = cm / total * 100

    # Combine counts and percentages for annotations
    annot = np.array([f"{cm[i,j]}\n({percentages[i,j]:.1f}%)"
                      for i in range(cm.shape[0]) for j in range(cm.shape[1])]).reshape(cm.shape)

    # Plot heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'],
                vmin=0, vmax=120)
    #plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.grid(False)
    plt.tight_layout()

    # Save as high-quality PDF
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Auto-download in Colab
    files.download(filename)

# Call the function
plot_cm_save_download(cm_novec, 'NOVEC Confusion Matrix', '/content/novec_confusion_matrix.pdf')

```

### Cell 113 — code (execution_count=None)
```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# --- Confusion Matrix ---
cm = confusion_matrix(y_test_old, voting_predictions)
percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Class labels
class_names = ['Negative', 'Positive']

# --- Plot ---
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title('Confusion Matrix - Voting Classifier')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Add count + percentage inside cells
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

# --- Save as High-Quality PDF (300 DPI) ---
pdf_path = "/content/voting_confusion_matrix.pdf"
plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')

plt.show()

# --- Auto-download the file in Colab ---
files.download(pdf_path)

```

### Cell 114 — code (execution_count=None)
```python
# ============================================================
# NOVEC Voting Classifier (ys54)
# Full Pipeline: Random Search → Voting → Metrics → ROC → CM → Mean±SD → Plot
# Satisfies Reviewer #2 Concern #19
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# 1. Random Search for Base Models
# ------------------------------------------------------------

# Logistic Regression
ys54_param_log = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']
}
ys54_log_rs = RandomizedSearchCV(LogisticRegression(), ys54_param_log, cv=5, n_iter=10,
                                 scoring='accuracy', random_state=42)
ys54_log_rs.fit(X_train_old, y_train_old)
ys54_logistic_best = ys54_log_rs.best_estimator_

# Random Forest
ys54_param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [10, 50, 100],
    'criterion': ['gini', 'entropy']
}
ys54_rf_rs = RandomizedSearchCV(RandomForestClassifier(), ys54_param_rf, cv=5, n_iter=10,
                                scoring='accuracy', random_state=42)
ys54_rf_rs.fit(X_train_old, y_train_old)
ys54_rf_best = ys54_rf_rs.best_estimator_

# KNN
ys54_param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean']
}
ys54_knn_rs = RandomizedSearchCV(KNeighborsClassifier(), ys54_param_knn, cv=5, n_iter=10,
                                 scoring='accuracy', random_state=42)
ys54_knn_rs.fit(X_train_old, y_train_old)
ys54_knn_best = ys54_knn_rs.best_estimator_

# SVM
ys54_param_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
ys54_svm_rs = RandomizedSearchCV(SVC(probability=True), ys54_param_svm, cv=5, n_iter=10,
                                 scoring='accuracy', random_state=42)
ys54_svm_rs.fit(X_train_old, y_train_old)
ys54_svm_best = ys54_svm_rs.best_estimator_


# ------------------------------------------------------------
# 2. NOVEC Voting Classifier (Soft Voting)
# ------------------------------------------------------------

ys54_voting = VotingClassifier(
    estimators=[
        ('knn', ys54_knn_best),
        ('rf', ys54_rf_best),
        ('log', ys54_logistic_best),
        ('svm', ys54_svm_best)
    ],
    voting='soft',
    weights=[4, 1, 1, 4]
)

ys54_voting.fit(X_train_old, y_train_old)

# Predictions
ys54_pred = ys54_voting.predict(X_test_old)


# ------------------------------------------------------------
# 3. Training & Test Metrics
# ------------------------------------------------------------

ys54_train_acc = ys54_voting.score(X_train_old, y_train_old)
ys54_test_acc = accuracy_score(y_test_old, ys54_pred)
ys54_prec = precision_score(y_test_old, ys54_pred)
ys54_rec = recall_score(y_test_old, ys54_pred)
ys54_f1 = f1_score(y_test_old, ys54_pred)

ys54_prob = ys54_voting.predict_proba(X_test_old)[:, 1]
fpr_ys54, tpr_ys54, _ = roc_curve(y_test_old, ys54_prob)
ys54_auc = auc(fpr_ys54, tpr_ys54)

print("\n========== NOVEC Voting Classifier (ys54) ==========")
print(f"Training Accuracy: {ys54_train_acc:.4f}")
print(f"Testing Accuracy : {ys54_test_acc:.4f}")
print(f"Precision        : {ys54_prec:.4f}")
print(f"Recall           : {ys54_rec:.4f}")
print(f"F1 Score         : {ys54_f1:.4f}")
print(f"AUC-ROC          : {ys54_auc:.4f}")
print("===================================================\n")


# ------------------------------------------------------------
# 4. ROC Curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(fpr_ys54, tpr_ys54, label=f'AUC = {ys54_auc:.4f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve – NOVEC Voting Classifier (ys54)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# 5. Confusion Matrix (Counts + Percentages)
# ------------------------------------------------------------

ys54_cm = confusion_matrix(y_test_old, ys54_pred)
ys54_cm_percent = ys54_cm.astype(float) / ys54_cm.sum(axis=1)[:, None] * 100

plt.figure(figsize=(6, 5))
sns.heatmap(
    ys54_cm, annot=[[f"{ys54_cm[i,j]}\n({ys54_cm_percent[i,j]:.1f}%)"
                     for j in range(ys54_cm.shape[1])] for i in range(ys54_cm.shape[0])],
    fmt="", cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – NOVEC Voting (ys54)")
plt.show()


# ------------------------------------------------------------
# 6. Mean ± SD (Reviewer #2 Concern #19)
# ------------------------------------------------------------

ys54_scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

ys54_cv = cross_validate(
    ys54_voting, X_train_old, y_train_old,
    cv=5, scoring=ys54_scoring
)

print("===== NOVEC Voting Classifier (ys54) – Mean ± SD =====")
for metric in ys54_scoring:
    scores = ys54_cv[f"test_{metric}"]
    print(f"{metric.capitalize():<10}: {scores.mean():.4f} ± {scores.std():.4f}")
print("=====================================================\n")


# ------------------------------------------------------------
# 7. Mean ± SD Plot
# ------------------------------------------------------------

ys54_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
ys54_means = [
    ys54_cv['test_accuracy'].mean(),
    ys54_cv['test_precision'].mean(),
    ys54_cv['test_recall'].mean(),
    ys54_cv['test_f1'].mean(),
    ys54_cv['test_roc_auc'].mean()
]

ys54_stds = [
    ys54_cv['test_accuracy'].std(),
    ys54_cv['test_precision'].std(),
    ys54_cv['test_recall'].std(),
    ys54_cv['test_f1'].std(),
    ys54_cv['test_roc_auc'].std()
]

x = np.arange(len(ys54_metrics))

plt.figure(figsize=(10, 6))
plt.errorbar(x, ys54_means, yerr=ys54_stds,
             fmt='o-', capsize=6, markersize=8, linewidth=2,
             color='darkgreen', ecolor='lightgreen')
plt.xticks(x, ys54_metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Mean ± SD – NOVEC Voting Classifier (ys54)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

```

### Cell 115 — markdown (execution_count=None)
```text
# Mean ± SD for Base Classifiers Only (Reviewer #2, Concern #19)
# Base Models: Logistic, Random Forest, KNN, SVM
```

### Cell 116 — code (execution_count=None)
```python
# ============================================================
# Mean ± SD for Base Classifiers Only (Reviewer #2 – Concern #19)
# Base Models: Logistic, Random Forest, KNN, SVM
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate

# ------------------------------------------------------------
# 1. Hyperparameter Search for Base Models
# ------------------------------------------------------------

# Logistic Regression
param_log = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']
}
log_rs = RandomizedSearchCV(
    LogisticRegression(),
    param_log, cv=5, n_iter=10,
    scoring='accuracy', random_state=42
)
log_rs.fit(X_train_old, y_train_old)
best_log = log_rs.best_estimator_

# Random Forest
param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [10, 50, 100],
    'criterion': ['gini', 'entropy']
}
rf_rs = RandomizedSearchCV(
    RandomForestClassifier(),
    param_rf, cv=5, n_iter=10,
    scoring='accuracy', random_state=42
)
rf_rs.fit(X_train_old, y_train_old)
best_rf = rf_rs.best_estimator_

# KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['manhattan', 'euclidean']
}
knn_rs = RandomizedSearchCV(
    KNeighborsClassifier(),
    param_knn, cv=5, n_iter=10,
    scoring='accuracy', random_state=42
)
knn_rs.fit(X_train_old, y_train_old)
best_knn = knn_rs.best_estimator_

# SVM
param_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
svm_rs = RandomizedSearchCV(
    SVC(probability=True),
    param_svm, cv=5, n_iter=10,
    scoring='accuracy', random_state=42
)
svm_rs.fit(X_train_old, y_train_old)
best_svm = svm_rs.best_estimator_

# ------------------------------------------------------------
# 2. Compute Mean ± SD of Metrics for Each Base Classifier
# ------------------------------------------------------------

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

base_models = {
    "Logistic Regression": best_log,
    "Random Forest": best_rf,
    "KNN": best_knn,
    "SVM": best_svm
}

print("\n================ Base Classifiers: Mean ± SD (5-Fold CV) ================\n")

for name, model in base_models.items():
    cv_results = cross_validate(
        model, X_train_old, y_train_old,
        cv=5, scoring=scoring
    )
    print(f"\n{name}:")
    for metric in scoring:
        mean = cv_results[f"test_{metric}"].mean()
        std = cv_results[f"test_{metric}"].std()
        print(f"  {metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")

print("\n=======================================================================\n")

```

### Cell 117 — markdown (execution_count=None)
```text
# Calibration (slope, intercept, Brier) and PR-AUC :::: Reviewer#2, Concern # 13:
```

### Cell 118 — code (execution_count=None)
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
novec_prob = voting_model.predict_proba(X_test_old)[:, 1]

# True labels
y_true = y_test_old

# ------------------------------------------------------------
# 2. Brier Score (Calibration Quality)
# ------------------------------------------------------------
brier = brier_score_loss(y_true, novec_prob)
print(f"Brier Score (NOVEC): {brier:.4f}")

# ------------------------------------------------------------
# 3. Calibration Slope & Calibration Intercept
#    Fit linear regression of:
#       predicted_probabilities (x) → actual outcomes (y)
# ------------------------------------------------------------

# Reshape probabilities for regression
novec_prob_reshaped = novec_prob.reshape(-1, 1)

# Linear regression: y_true = intercept + slope * predicted_prob
lr = LinearRegression().fit(novec_prob_reshaped, y_true)

calib_intercept = lr.intercept_
calib_slope = lr.coef_[0]

print(f"Calibration Intercept (NOVEC): {calib_intercept:.4f}")
print(f"Calibration Slope (NOVEC):     {calib_slope:.4f}")

# ------------------------------------------------------------
# 4. PR-AUC (Precision–Recall AUC)
# ------------------------------------------------------------

precision, recall, _ = precision_recall_curve(y_true, novec_prob)

# PR-AUC using trapezoidal integration
pr_auc = auc(recall, precision)

# AP score (area under PR curve using average precision)
ap_score = average_precision_score(y_true, novec_prob)

print(f"PR-AUC (NOVEC):                 {pr_auc:.4f}")
print(f"Average Precision Score (NOVEC): {ap_score:.4f}")

# ------------------------------------------------------------
# Optional: Plot Calibration Curve
# ------------------------------------------------------------

import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_true, novec_prob, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker='o', label='NOVEC Calibration')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve – NOVEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.show()

```

### Cell 119 — code (execution_count=None)
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


# ----------------------------------------------------------
# 1. Train Base Models
# ----------------------------------------------------------

# Logistic Regression
param_log = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']
}
log_rs = RandomizedSearchCV(LogisticRegression(), param_log, n_iter=10, cv=5, scoring='accuracy', random_state=42)
log_rs.fit(X_train_old, y_train_old)
logistic_best_rs = log_rs.best_estimator_

# Random Forest
param_rf = {
    'max_depth': [2, 5, None],
    'n_estimators': [2],
    'criterion': ['entropy']
}
rf_rs = RandomizedSearchCV(RandomForestClassifier(), param_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)
rf_rs.fit(X_train_old, y_train_old)
rf_best_rs = rf_rs.best_estimator_

# KNN
param_knn = {
    'n_neighbors': [5, 7, 9, 11],
    'weights': ['distance'],
    'metric': ['manhattan']
}
knn_rs = RandomizedSearchCV(KNeighborsClassifier(), param_knn, n_iter=10, cv=5, scoring='accuracy', random_state=42)
knn_rs.fit(X_train_old, y_train_old)
knn_best_rs = knn_rs.best_estimator_

# SVM
param_svm = {
    'C': [100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']
}
svm_rs = RandomizedSearchCV(SVC(probability=True), param_svm, n_iter=10, cv=5, scoring='accuracy', random_state=42)
svm_rs.fit(X_train_old, y_train_old)
svm_best_rs = svm_rs.best_estimator_


# ----------------------------------------------------------
# 2. Create Voting Classifier
# ----------------------------------------------------------

voting_model = VotingClassifier(
    estimators=[
        ('knn', knn_best_rs),
        ('rf', rf_best_rs),
        ('logistic', logistic_best_rs),
        ('svm', svm_best_rs)
    ],
    voting='soft',
    weights=[4, 1, 1, 4]
)

voting_model.fit(X_train_old, y_train_old)

# ----------------------------------------------------------
# 3. Apply Platt Scaling (Sigmoid Calibration)
# ----------------------------------------------------------

platt_calibrated_model = CalibratedClassifierCV(
    estimator=voting_model,
    method='sigmoid',
    cv='prefit'
)

platt_calibrated_model.fit(X_train_old, y_train_old)


# ----------------------------------------------------------
# 4. Predictions & Core Metrics
# ----------------------------------------------------------

voting_prob_platt = platt_calibrated_model.predict_proba(X_test_old)[:, 1]
voting_pred_platt = (voting_prob_platt >= 0.5).astype(int)

train_accuracy = voting_model.score(X_train_old, y_train_old)
test_accuracy = accuracy_score(y_test_old, voting_pred_platt)
precision = precision_score(y_test_old, voting_pred_platt)
recall = recall_score(y_test_old, voting_pred_platt)
f1 = f1_score(y_test_old, voting_pred_platt)

# ROC AUC
fpr, tpr, thresholds = roc_curve(y_test_old, voting_prob_platt)
auc_score = auc(fpr, tpr)

# PR-AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test_old, voting_prob_platt)
pr_auc = auc(recall_curve, precision_curve)

# Average Precision Score
avg_precision = average_precision_score(y_test_old, voting_prob_platt)

# Brier Score
brier = brier_score_loss(y_test_old, voting_prob_platt)


# ----------------------------------------------------------
# 4A. Calibration Slope & Intercept
# ----------------------------------------------------------

probs_reshaped = voting_prob_platt.reshape(-1, 1)
calib_lr = LogisticRegression().fit(probs_reshaped, y_test_old)

calib_slope = calib_lr.coef_[0][0]
calib_intercept = calib_lr.intercept_[0]


# ----------------------------------------------------------
# 5. Print All Metrics
# ----------------------------------------------------------

print("========== Platt-Calibrated NOVEC Model ==========")
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
plt.plot(fpr, tpr, color='blue', label=f'Calibrated NOVEC AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve – Platt-Calibrated NOVEC Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 7. Plot Calibration Curve
# ----------------------------------------------------------

prob_true, prob_pred = calibration_curve(y_test_old, voting_prob_platt, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated NOVEC')
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
baseline = np.mean(y_test_old)
plt.hlines(baseline, xmin=0, xmax=1, colors='gray', linestyles='--', label=f'Baseline = {baseline:.3f}')

plt.title("Precision–Recall Curve – Platt-Calibrated NOVEC Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()


```

### Cell 120 — code (execution_count=None)
```python
# ----------------------------------------------------------
# 4. Predictions & Core Metrics
# ----------------------------------------------------------

voting_prob_platt = platt_calibrated_model.predict_proba(X_test_old)[:, 1]
voting_pred_platt = (voting_prob_platt >= 0.5).astype(int)

train_accuracy = voting_model.score(X_train_old, y_train_old)
test_accuracy = accuracy_score(y_test_old, voting_pred_platt)
precision = precision_score(y_test_old, voting_pred_platt)
recall = recall_score(y_test_old, voting_pred_platt)
f1 = f1_score(y_test_old, voting_pred_platt)

# ROC AUC
fpr, tpr, thresholds = roc_curve(y_test_old, voting_prob_platt)
auc_score = auc(fpr, tpr)

# PR-AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test_old, voting_prob_platt)
pr_auc = auc(recall_curve, precision_curve)

# Average Precision Score
avg_precision = average_precision_score(y_test_old, voting_prob_platt)

# Brier Score
brier = brier_score_loss(y_test_old, voting_prob_platt)


# ----------------------------------------------------------
# 4A. Calibration Slope & Intercept
# ----------------------------------------------------------

probs_reshaped = voting_prob_platt.reshape(-1, 1)
calib_lr = LogisticRegression().fit(probs_reshaped, y_test_old)

calib_slope = calib_lr.coef_[0][0]
calib_intercept = calib_lr.intercept_[0]


# ----------------------------------------------------------
# 5. Print All Metrics
# ----------------------------------------------------------

print("========== Platt-Calibrated NOVEC Model ==========")
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
plt.plot(fpr, tpr, color='blue', label=f'Calibrated NOVEC AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve – Platt-Calibrated NOVEC Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 7. Plot Calibration Curve
# ----------------------------------------------------------

prob_true, prob_pred = calibration_curve(y_test_old, voting_prob_platt, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated NOVEC')
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
baseline = np.mean(y_test_old)
plt.hlines(baseline, xmin=0, xmax=1, colors='gray', linestyles='--', label=f'Baseline = {baseline:.3f}')

plt.title("Precision–Recall Curve – Platt-Calibrated NOVEC Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.show()

```

### Cell 121 — code (execution_count=None)
```python
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# ----------------------------------------------------------
# 7. Plot Calibration Curve (Save as PDF 300 DPI)
# ----------------------------------------------------------

plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated Proposed OVEC')
plt.plot([0, 1], [0, 1], '--', color='black', label='Perfect Calibration')
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
         label=f"PR-AUC of Platt-Calibrated Proposed OVEC  = {pr_auc:.4f}")

baseline = np.mean(y_test_old)
plt.hlines(baseline, xmin=0, xmax=1, colors='black', linestyles='--',
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

### Cell 122 — code (execution_count=None)
```python

```

### Cell 123 — code (execution_count=None)
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

y_true = y_test_old

voting_prob_platt = platt_calibrated_model.predict_proba(X_test_old)[:, 1]
voting_pred_platt = (voting_prob_platt >= 0.5).astype(int)


y_prob = voting_prob_platt   # shape (n_samples,)


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

### Cell 124 — code (execution_count=None)
```python
from google.colab import files

# ------------------------------------------------------------
# 4. Plot & Save DCA Curve (300 DPI PDF + Auto-Download)
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(thresholds, net_benefit_novec, label="Proposed Optimized Voting Ensemble Classifier", linewidth=2.1)
plt.plot(thresholds, nb_treat_none, '--', label="Treat None", linewidth=1.8)
plt.plot(thresholds, nb_treat_all, '--', label="Treat All", linewidth=1.8)

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
#plt.title("Decision Curve Analysis (DCA) – OVEC Voting Classifier")
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

### Cell 125 — code (execution_count=None)
```python

```

### Cell 126 — code (execution_count=None)
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

### Cell 127 — code (execution_count=None)
```python

```

### Cell 128 — code (execution_count=None)
```python

```

### Cell 129 — code (execution_count=None)
```python
# ============================================================
# 5. Decision Curve Analysis (DCA) for NOVEC Voting Classifier
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def decision_curve_analysis(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 50)):
    """
    Compute net benefit for a binary classifier across a range
    of threshold probabilities using standard DCA formula.

    y_true : array-like of shape (n_samples,)
        True binary labels (0/1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    thresholds : array-like
        Threshold probabilities at which to calculate net benefit.
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
        nb = (tp / n) - (fp / n) * (t / (1.0 - t))
        net_benefit.append(nb)

    return thresholds, np.array(net_benefit)

# Run DCA for NOVEC
thresholds, net_benefit_novec = decision_curve_analysis(y_true, novec_prob)

# "Treat None" strategy → net benefit = 0 for all thresholds
nb_treat_none = np.zeros_like(thresholds)

# "Treat All" strategy → simplified net benefit using outcome prevalence
prevalence = np.mean(y_true)
nb_treat_all = prevalence - (1 - prevalence) * (thresholds / (1.0 - thresholds))

# ------------------------------------------------------------
# Plot Decision Curve Analysis
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.plot(thresholds, net_benefit_novec, label="NOVEC Voting Classifier")
plt.plot(thresholds, nb_treat_none, linestyle="--", label="Treat None")
plt.plot(thresholds, nb_treat_all, linestyle="--", label="Treat All")

plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis – NOVEC Voting Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

### Cell 130 — code (execution_count=None)
```python

```

### Cell 131 — code (execution_count=None)
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
evaluate_model(logistic_best_rs, X_train_old, y_train_old, X_test_old, y_test_old, name="Logistic Regression")
evaluate_model(rf_rs, X_train_old, y_train_old, X_test_old, y_test_old, name="Random Forest")
evaluate_model(knn_best_rs, X_train_old, y_train_old, X_test_old, y_test_old, name="K-Nearest Neighbors")
evaluate_model(svm_best_rs, X_train_old, y_train_old, X_test_old, y_test_old, name="Support Vector Machine")

```

### Cell 132 — code (execution_count=None)
```python

```

### Cell 133 — code (execution_count=None)
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

### Cell 134 — code (execution_count=None)
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

### Cell 135 — code (execution_count=None)
```python
from google.colab import files
files.download("ROC_NOVEC_NOSEC.pdf")

```

### Cell 136 — code (execution_count=None)
```python

```

### Cell 137 — code (execution_count=None)
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

### Cell 138 — code (execution_count=None)
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

### Cell 139 — code (execution_count=None)
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

### Cell 140 — code (execution_count=None)
```python
from google.colab import files
files.download("ROC_NOVEC_NOSEC.pdf")
```

### Cell 141 — code (execution_count=None)
```python

```

### Cell 142 — code (execution_count=None)
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

### Cell 143 — markdown (execution_count=None)
```text
# paired Wilcoxon across folds) and CIs for AUC/F1.
```

### Cell 144 — code (execution_count=None)
```python
# Step 1 — Run 5-fold CV for BOTH models (OVEC & OSEC)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

### Cell 145 — code (execution_count=None)
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

### Cell 146 — code (execution_count=None)
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