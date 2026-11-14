# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\five_states_results\Five States Results\OSEC_5state_dataset_Clinical_Utility_Analysis.ipynb`
- SHA-256: `dca8d66571a0cb304d399ebe6e81bf2c9b99682788763d3f0a31bd1175f4f097`
- Cells: 63
- Saved outputs: 57

## Imports
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
- `xgboost`

## Method keywords
- CalibratedClassifierCV
- GridSearchCV
- LoRAS
- SMOTE
- StackingClassifier
- bootstrap
- train_test_split

## Risk flags
- **credential_or_mount**: cells [1]
- **notebook_shell**: cells [7, 17]

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
##Algorithms Without hyper parameter-tuning
```

### Cell 23 — code (execution_count=None)
```python
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Create base estimators with default parameters (no hyperparameter tuning)
bayes = GaussianNB()
adaboost = AdaBoostClassifier(random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)

# Create the Stacking Classifier using default estimators
stacking_classifier = StackingClassifier(
    cv=5,
    estimators=[
        ('bayes', bayes),
        ('Adaboost', adaboost),
        ('Decision tree', decision_tree),
        ('Logistic Regression', logistic_regression)
    ],
    final_estimator=DecisionTreeClassifier(random_state=42)
)

# Calibrate the Stacking Classifier
calibrated_stacking = CalibratedClassifierCV(stacking_classifier, method='sigmoid', cv=5)

# Train the calibrated stacking classifier
print("Training the Stacking Classifier...")
calibrated_stacking.fit(X_train, y_train)

# Make predictions
y_pred = calibrated_stacking.predict(X_test)
y_pred_proba = calibrated_stacking.predict_proba(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # Use 'weighted' for multiclass
recall = recall_score(y_test, y_pred, average='binary')  # Use 'weighted' for multiclass
f1 = f1_score(y_test, y_pred, average='binary')  # Use 'weighted' for multiclass
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])  # For binary classification

# Display results
print("\n" + "="*60)
print("STACKING CLASSIFIER PERFORMANCE METRICS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print("="*60)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Cell 24 — markdown (execution_count=None)
```text
# **DT**
```

### Cell 25 — code (execution_count=None)
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

### Cell 26 — code (execution_count=None)
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

### Cell 28 — markdown (execution_count=None)
```text
# **Gradient Boosting**
```

### Cell 29 — code (execution_count=None)
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

### Cell 30 — markdown (execution_count=None)
```text
# **Random Forest**


```

### Cell 31 — code (execution_count=None)
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

### Cell 32 — code (execution_count=None)
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

### Cell 33 — markdown (execution_count=None)
```text
# **Logistic Regression**
```

### Cell 34 — code (execution_count=None)
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

### Cell 37 — markdown (execution_count=None)
```text
# **AdaBoost**
```

### Cell 38 — code (execution_count=None)
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

### Cell 39 — markdown (execution_count=None)
```text
# **AdaBoost-SVM**
```

### Cell 40 — code (execution_count=None)
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

### Cell 43 — markdown (execution_count=None)
```text
# **XGBoost Regressor**
```

### Cell 44 — code (execution_count=None)
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

### Cell 45 — markdown (execution_count=None)
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

### Cell 46 — markdown (execution_count=None)
```text
# **Lightgbm**, *Reviewer 2, Concern 18*
```

### Cell 47 — code (execution_count=None)
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

### Cell 48 — markdown (execution_count=None)
```text
# SVM
```

### Cell 49 — code (execution_count=None)
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

### Cell 50 — markdown (execution_count=None)
```text
KNN
```

### Cell 51 — code (execution_count=None)
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

### Cell 52 — markdown (execution_count=None)
```text
# **ML Individual Comparison**
```

### Cell 53 — code (execution_count=None)
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
#compar.loc[len(compar)] = ['AdaBoost-SVM', test_accuracy_absvm]
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

### Cell 54 — markdown (execution_count=None)
```text
# **New Stacking for 04 base** *Reviewer 2, Concern 22*
```

### Cell 55 — code (execution_count=None)
```python
# Create the Stacking Classifier 1 using the best estimators from Grid Search
stacking_classifier_rf = StackingClassifier(
    cv=5,
    estimators=[
        ('bayes', best_bayes),
        ('Adaboost', best_ab),
        ('Decision tree', best_tree),
        ('Logistic Regression', best_logistic)
    ],
    final_estimator=best_tree
)

# Calibrate the Stacking Classifier
calibrated_stacking_rf = CalibratedClassifierCV(stacking_classifier_rf, method='sigmoid', cv=5)
calibrated_stacking_rf.fit(X_train, y_train)
```

### Cell 56 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import roc_curve, auc # Import the auc function
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fit the ys42 Classifier
calibrated_stacking_rf.fit(X_train, y_train)

# Make predictions
ys42_predictions = calibrated_stacking_rf.predict(X_test)

# Calculate metrics
train_accuracy_ys42 = calibrated_stacking_rf.score(X_train, y_train)
test_accuracy_ys42 = accuracy_score(y_test, ys42_predictions)
precision_ys42 = precision_score(y_test, ys42_predictions, average='binary')  # Adjust for multi-class if needed
recall_ys42 = recall_score(y_test, ys42_predictions, average='binary')  # Adjust for multi-class if needed
f1_ys42 = f1_score(y_test, ys42_predictions, average='binary')  # Adjust for multi-class if needed

# For AUC-ROC, calculate probabilities (soft ys42 would be needed for using probabilities)
if hasattr(calibrated_stacking_rf, "predict_proba"):
    ys42_probabilities = calibrated_stacking_rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    fpr_ys42, tpr_ys42, thresholds = roc_curve(y_test, ys42_probabilities)
    auc_score = auc(fpr_ys42, tpr_ys42)

    # Plot the AUC-ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_ys42, tpr_ys42, color='blue', label=f'Stacking Classifier AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
else:
    auc_score = "AUC-ROC not available with hard ys42."

# Display all metrics
print(f'Training Accuracy_ys42: {train_accuracy_ys42:.4f}')
print(f'Testing Accuracy_ys42: {test_accuracy_ys42:.4f}')
print(f'Precision_ys42: {precision_ys42:.4f}')
print(f'Recall_ys42: {recall_ys42:.4f}')
print(f'F1_ys42 Score: {f1_ys42:.4f}')
print(f'AUC-ROC_ys42: {auc_score}')


# Compute confusion matrix
cm_ys42 = confusion_matrix(y_test, ys42_predictions)

# Calculate percentages
cm_ys42_percent = cm_ys42.astype('float') / cm_ys42.sum(axis=1)[:, np.newaxis] * 100

# Labels with counts and percentages
labels = np.array([[f"{count}\n({percent:.1f}%)" for count, percent in zip(row, percent_row)]
                   for row, percent_row in zip(cm_ys42, cm_ys42_percent)])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ys42, annot=labels, fmt="", cmap="Blues", cbar=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Counts & Percentages)")
plt.show()
```

### Cell 57 — code (execution_count=None)
```python
import seaborn as sns
compar=pd.DataFrame(columns=["algorithm","Accuracy"])
compar.loc[len(compar)] = ['LogisticRegression', test_accuracy_logistic]
compar.loc[len(compar)] = ['DecisionTreeClassifier', test_accuracy]
compar.loc[len(compar)] = ['lightgbm', test_accuracy_lgbm]
compar.loc[len(compar)] = ['Bayesian Networks', test_accuracy_bayes]
compar.loc[len(compar)] = ['AdaBoost', test_accuracy_ab]
compar.loc[len(compar)] = ['XGBoost Classifier', test_accuracy_xgb]
compar.loc[len(compar)] = ['Proposed_Stacking', test_accuracy_ys42]
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

### Cell 58 — markdown (execution_count=None)
```text
# **Confidence Intervals** *Reviewer 2, Concern 19*
```

### Cell 59 — code (execution_count=None)
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
y_pred = calibrated_stacking_rf.predict(X_test)
y_proba = calibrated_stacking_rf.predict_proba(X_test)[:, 1]

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

    y_pred_resampled = calibrated_stacking_rf.predict(X_resampled)
    y_proba_resampled = calibrated_stacking_rf.predict_proba(X_resampled)[:, 1]

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

### Cell 60 — markdown (execution_count=None)
```text

```

### Cell 61 — code (execution_count=None)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# -----------------------------
# 1. Get model predictions (Calibrated Stacking Classifier)
# -----------------------------
y_probs = calibrated_stacking_rf.predict_proba(X_test)[:, 1]  # Probabilities for positive class
y_pred = calibrated_stacking_rf.predict(X_test)

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

### Cell 62 — code (execution_count=None)
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
    "Proposed Stacking": calibrated_stacking_rf    # your final stacked calibrated model
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

### Cell 63 — code (execution_count=None)
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
    "Proposed Stacking": calibrated_stacking_rf    # your final stacked calibrated model
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