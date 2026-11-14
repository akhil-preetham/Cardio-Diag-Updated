# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\mendeley_results\MENDELEY Results\Fairness_OSEC_MENDELEY_dataset.ipynb`
- SHA-256: `da97e4ed1d39a33e1c0587dfafb54998c3c0b14ec40e2165c4836b6c551c2e01`
- Cells: 68
- Saved outputs: 85

## Imports
- `google.colab`
- `imblearn.over_sampling`
- `imblearn.pipeline`
- `loras`
- `matplotlib.patches`
- `matplotlib.pyplot`
- `numpy`
- `openpyxl`
- `os`
- `pandas`
- `seaborn`
- `sklearn.calibration`
- `sklearn.compose`
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
- `time`
- `warnings`
- `xgboost`

## Method keywords
- CalibratedClassifierCV
- GridSearchCV
- LoRAS
- SMOTE
- StackingClassifier
- VotingClassifier
- bootstrap
- train_test_split

## Risk flags
- **credential_or_mount**: cells [1, 39, 60, 68]
- **notebook_shell**: cells [18]

## Cell sources

### Cell 1 — code (execution_count=2)
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

### Cell 2 — code (execution_count=3)
```python
mendeley = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/MendeleyCardiovascular_Disease_Dataset.csv')

#first 5 values of data
mendeley.head()

```

### Cell 3 — code (execution_count=4)
```python
# Check the result
print("Data as NaN:")
print(mendeley.isnull().sum())
print("Data Shape:", mendeley.shape)
```

### Cell 4 — code (execution_count=5)
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Cell 5 — code (execution_count=6)
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

### Cell 6 — code (execution_count=7)
```python
# Drop Unwanted Columns
mendeley.drop(columns=["patientid", "noofmajorvessels"], inplace=True, errors="ignore")
mendeley.shape

```

### Cell 7 — code (execution_count=8)
```python
print("Mendeley ST_Slope:", sorted(mendeley["ST_Slope"].unique()))
```

### Cell 8 — code (execution_count=9)
```python
# Step 1️⃣ Remove ST_Slope == 0
mendeley= mendeley[mendeley["ST_Slope"] != 0].reset_index(drop=True)
print("Mendeley ST_Slope:", sorted(mendeley["ST_Slope"].unique()))
```

### Cell 9 — code (execution_count=10)
```python
mendeley["ST_Slope"] = mendeley["ST_Slope"] - 1
print("Mendeley ST_Slope:", sorted(mendeley["ST_Slope"].unique()))
```

### Cell 10 — code (execution_count=11)
```python
X = mendeley.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = mendeley['HeartDisease']
print(mendeley)
print("X",X)
print("y",y)

#target Variables Values
mendeley.HeartDisease.value_counts()
```

### Cell 11 — code (execution_count=12)
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

### Cell 12 — code (execution_count=13)
```python
X = mendeley_no_outliers.drop('HeartDisease', axis=1)  # Assuming 'HeartDisease' is the label column
y = mendeley_no_outliers['HeartDisease']


#HeartDisease Variables Values
mendeley_no_outliers.HeartDisease.value_counts()
X.shape
```

### Cell 13 — code (execution_count=14)
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

### Cell 14 — code (execution_count=15)
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

### Cell 15 — code (execution_count=16)
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

### Cell 16 — code (execution_count=17)
```python
df_imputed_X = df_imputed.drop('HeartDisease', axis=1)  # Assuming 'HeartDisease' is the label column
df_imputed_y = df_imputed['HeartDisease']
print(df_imputed_X.shape)
```

### Cell 17 — code (execution_count=18)
```python
# ! pip install smote_variants==0.5.1
# ! pip install smote-variants
```

### Cell 18 — code (execution_count=19)
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

### Cell 19 — code (execution_count=20)
```python
# 2️⃣ Apply LoRAS
df_imputed_X_minority_loras = loras.fit_resample( df_imputed_X_majority, df_imputed_X_minority)
print("After LoRAS:")
print("Minority class:", df_imputed_X_minority_loras.shape)
print("Majority class:", df_imputed_X_majority.shape)
```

### Cell 20 — code (execution_count=21)
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

### Cell 21 — code (execution_count=22)
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

### Cell 22 — code (execution_count=23)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming you have your data loaded into X (features) and y (labels)

# Apply PCA to reduce the dimensionality to 9
pca = PCA(n_components=9)
X_oversampled_LoRAS_features_pca9 = pca.fit_transform(X_loras)
```

### Cell 23 — code (execution_count=24)
```python
# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled_LoRAS_features_pca9, y_loras, test_size=0.25, random_state=42)

# Step 9: Print shapes of the resulting datasets
print(f'Train features shape: {X_train.shape}')
print(f'Test features shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}')
print(f'Test labels shape: {y_test.shape}')
```

### Cell 24 — code (execution_count=25)
```python
# import pandas as pd
# import numpy as np
# import smote_variants as v
# oversampler= v.ProWSyn()

# # Convert the DataFrame to a NumPy array
# X_array = df_imputed_X.to_numpy()  # or use df_imputed_X.values
# y_array = df_imputed_y.to_numpy()

# # X_samp and y_samp contain the oversampled dataset
# X_oversampled_prowsyn , y_oversampled_prowsyn = oversampler.sample(X_array,y_array)

# # # Assuming you have a NumPy array 'numpy_array' and a pandas DataFrame 'df'
# X_oversampled_prowsyn_df = pd.DataFrame(X_oversampled_prowsyn)
# # y_oversampled_df = pd.DataFrame(y_oversampled)
# # # concatenated_df = pd.concat([df, numpy_df], axis=1)

# # data_oversampled = pd.concat([X_oversampled_df , y_oversampled_df], axis=1)
# # data_oversampled.shape
# # #Target Variables Values
# # # data_oversampled.HeartDiseaseorAttack.value_counts()
# #print("X_oversampled_prowsyn",X_oversampled_prowsyn)
# print("X_oversampled_prowsyn.shape", X_oversampled_prowsyn.shape)
# #print("y_oversampled_prowsyn",y_oversampled_prowsyn)
# print("y_oversampled_prowsyn.shape", y_oversampled_prowsyn.shape)
```

### Cell 25 — code (execution_count=26)
```python
# # Use np.bincount to count occurrences of each value
# # Convert y_oversampled_prowsyn to int before using np.bincount
# label_counts_prowsyn = np.bincount(y_oversampled_prowsyn.astype(int))
# print("label_counts_prowsyn.shape",label_counts_prowsyn.shape)
# print("label_counts_prowsyn",label_counts_prowsyn)
```

### Cell 26 — code (execution_count=27)
```python
# import matplotlib.pyplot as plt

# # Assuming label_counts_prowsyn is a NumPy array with shape (2,) containing [244831, 214715]
# labels = ['No Heart Disease', 'Heart Disease']  # Label names
# counts_prowsyn = label_counts_prowsyn  # Corresponding counts_prowsyn

# # Create a bar graph
# plt.bar(labels, counts_prowsyn,  width=0.8, edgecolor='black', color=['blue', 'green'])

# # Add labels to the bars
# for i, count in enumerate(counts_prowsyn):
#     plt.text(i, count, str(count), ha='center', va='bottom')

# # Set the x-axis and y-axis labels.
# plt.xlabel('Heart Disease')
# plt.ylabel('Counts')
# # Add a title to the plot.
# plt.title('Distribution of Heart Disease by proWSYN')

# # Add the first legend.
# plt.legend(['No Heart Disease', 'Heart Disease'], loc='upper right')

# # Display the plot.
# plt.show()
```

### Cell 27 — code (execution_count=28)
```python
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Assuming you have your data loaded into X (features) and y (labels)
# # X and y should have shapes (253661, 21) and (253661,), respectively

# # Apply PCA to reduce the dimensionality to 2
# pca = PCA(n_components=9)
# X_oversampled_prowsyn_pca9 = pca.fit_transform(X_oversampled_prowsyn)
```

### Cell 28 — code (execution_count=29)
```python
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_oversampled_prowsyn_pca9)

# # Get the loadings (components)
# loadings = pca.components_

# # Create a DataFrame to visualize the loadings
# loading_df = pd.DataFrame(loadings, columns=X.columns, index=[f'PC{i+1}' for i in range(9)])

# # Display the loadings
# print(loading_df)
```

### Cell 29 — code (execution_count=30)
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

### Cell 30 — code (execution_count=31)
```python
# # Step 8: Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_oversampled_prowsyn_pca9 , y_oversampled_prowsyn, test_size=0.2, random_state=42)

# # Step 9: Print shapes of the resulting datasets
# print(f'Train features shape: {X_train.shape}')
# print(f'Test features shape: {X_test.shape}')
# print(f'Train labels shape: {y_train.shape}')
# print(f'Test labels shape: {y_test.shape}')
```

### Cell 31 — code (execution_count=32)
```python
# pip install xgboost==1.6.2

```

### Cell 32 — code (execution_count=33)
```python
# pip install --upgrade scikit-learn xgboost

```

### Cell 33 — code (execution_count=34)
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

### Cell 34 — code (execution_count=35)
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

### Cell 35 — code (execution_count=None)
```python

```

### Cell 36 — code (execution_count=None)
```python
# ============================================================
# Comprehensive Subgroup / Fairness Analysis for Heart Disease
# Fixed PCA transformation (numeric features only)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Inputs:
# - mendeley: full mendeleyFrame after preprocessing
# - voting_model: trained ensemble classifier on PCA-transformed features
# - pca: fitted PCA object from training (n_components=9)
# -----------------------------

y_true_col = "HeartDisease"
threshold = 0.5  # classification threshold

# ============================================================
# Helper functions
# ============================================================

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1 - p))

def compute_metrics(y_true, y_score, threshold=0.5, n_boot=1000):
    res = {}
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # AUROC + bootstrap CI
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = np.nan

    boot = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_score[idx]
        try:
            boot.append(roc_auc_score(yt, yp))
        except:
            continue

    if len(boot) > 0:
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    else:
        ci_low = ci_high = np.nan

    res["auc"] = auc
    res["auc_ci_low"] = ci_low
    res["auc_ci_high"] = ci_high

    # Calibration slope & intercept
    try:
        X_logit = logit(y_score).reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(X_logit, y_true)
        res["calib_intercept"] = lr.intercept_[0]
        res["calib_slope"] = lr.coef_[0][0]
    except:
        res["calib_intercept"] = np.nan
        res["calib_slope"] = np.nan

    # Threshold metrics
    y_pred = (y_score >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except:
        tn = fp = fn = tp = np.nan

    safe = lambda n, d: np.nan if d == 0 else n / d
    res["sensitivity"] = safe(tp, tp + fn)
    res["specificity"] = safe(tn, tn + fp)
    res["ppv"] = safe(tp, tp + fp)
    res["npv"] = safe(tn, tn + fn)

    res["n"] = len(y_true)
    res["events"] = int(y_true.sum())
    res["event_rate"] = safe(res["events"], res["n"])
    res["threshold"] = threshold

    return res

def evaluate_by_group(df, group_col, threshold=0.5):
    rows = []
    for val, sub in df.groupby(group_col):
        m = compute_metrics(sub[y_true_col], sub[y_score_col], threshold)
        m["group_var"] = group_col
        m["group"] = val
        rows.append(m)
    return pd.DataFrame(rows)

# ============================================================
# Step 1: PCA transform the full mendeleyset (numeric features only)
# ============================================================

# Define numeric features used in PCA (must match training)
numeric_features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

X_full_numeric = mendeley[numeric_features].copy()
X_full_pca = pca.transform(X_full_numeric)  # Shape: (n_samples, 9)

# ============================================================
# Step 2: Generate predicted probabilities
# ============================================================

y_score_col = "PredictedRisk"
mendeley[y_score_col] = calibrated_stacking.predict_proba(X_full_pca)[:, 1]

# ============================================================
# Step 3: Construct meaningful subgroups
# ============================================================

# Age groups
mendeley["AgeGroup"] = pd.cut(
    mendeley["Age"], bins=[-np.inf, 40, 60, np.inf], labels=["<40", "40–60", ">60"]
)

# Blood pressure groups
mendeley["BP_Group"] = pd.cut(
    mendeley["RestingBP"], bins=[-np.inf, 120, 139, np.inf], labels=["Normal", "Elevated", "Hypertension"]
)

# Cholesterol groups
mendeley["Chol_Group"] = pd.cut(
    mendeley["Cholesterol"], bins=[-np.inf, 200, 239, np.inf], labels=["<200", "200–239", "≥240"]
)

# MaxHR groups
mendeley["MaxHR_Group"] = pd.cut(
    mendeley["MaxHR"], bins=[-np.inf, 120, 160, np.inf], labels=["<120", "120–160", ">160"]
)

# Oldpeak groups
mendeley["Oldpeak_Group"] = pd.cut(mendeley["Oldpeak"], bins=[-np.inf, 1, np.inf], labels=["0–1", ">1"])

# Subgroup variables
subgroup_vars = [
    "Sex", "AgeGroup", "ChestPainType", "BP_Group", "Chol_Group",
    "FastingBS", "RestingECG", "MaxHR_Group", "ExerciseAngina",
    "Oldpeak_Group", "ST_Slope"
]

# ============================================================
# Step 4: Compute overall metrics
# ============================================================

overall_metrics = compute_metrics(mendeley[y_true_col], mendeley[y_score_col], threshold)
overall_df = pd.DataFrame([overall_metrics])
overall_df.insert(0, "group_var", "overall")
overall_df.insert(1, "group", "overall")

# ============================================================
# Step 5: Compute subgroup metrics
# ============================================================

all_results = []
for var in subgroup_vars:
    res = evaluate_by_group(mendeley, var, threshold)
    all_results.append(res)

subgroup_results = pd.concat(all_results, ignore_index=True)

# ============================================================
# Step 6: Save results
# ============================================================

overall_df.to_csv("Overall_Performance.csv", index=False)
subgroup_results.to_csv("ALL_Subgroup_Performance.csv", index=False)

print("✅ Generated files:")
print(" - Overall_Performance.csv")
print(" - ALL_Subgroup_Performance.csv")


```

### Cell 37 — code (execution_count=None)
```python

```

### Cell 38 — code (execution_count=37)
```python
# ============================================================
# Subgroup / Fairness Analysis with Bootstrap CIs (Formatted for Paper)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

y_true_col = "HeartDisease"
threshold = 0.5
n_boot = 1000  # bootstrap iterations

# -----------------------------
# Helper functions
# -----------------------------

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1 - p))

def safe_div(n, d):
    return np.nan if d == 0 else n / d

def compute_metrics_with_ci(y_true, y_score, threshold=0.5, n_boot=1000):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    res = {}

    # AUROC
    try:
        res["auc"] = roc_auc_score(y_true, y_score)
    except:
        res["auc"] = np.nan

    # Calibration
    try:
        X_logit = logit(y_score).reshape(-1,1)
        lr = LogisticRegression()
        lr.fit(X_logit, y_true)
        res["calib_intercept"] = lr.intercept_[0]
        res["calib_slope"] = lr.coef_[0][0]
    except:
        res["calib_intercept"] = np.nan
        res["calib_slope"] = np.nan

    # Threshold metrics
    y_pred = (y_score >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except:
        tn = fp = fn = tp = np.nan

    metrics = ["sensitivity", "specificity", "ppv", "npv"]
    values = [safe_div(tp, tp+fn), safe_div(tn, tn+fp), safe_div(tp, tp+fp), safe_div(tn, tn+fn)]

    # Initialize result
    for m, v in zip(metrics, values):
        res[m] = v

    # Sample info
    res["n"] = len(y_true)
    res["events"] = int(y_true.sum())
    res["event_rate"] = safe_div(res["events"], res["n"])
    res["threshold"] = threshold

    # Bootstrap CI
    rng = np.random.RandomState(42)
    metrics_boot = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_score[idx]
        yb_pred = (yp >= threshold).astype(int)
        try:
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yb_pred, labels=[0,1]).ravel()
        except:
            continue
        metrics_boot["sensitivity"].append(safe_div(tp_b, tp_b+fn_b))
        metrics_boot["specificity"].append(safe_div(tn_b, tn_b+fp_b))
        metrics_boot["ppv"].append(safe_div(tp_b, tp_b+fp_b))
        metrics_boot["npv"].append(safe_div(tn_b, tn_b+fn_b))

    # Compute CI and format
    for m in metrics:
        ci_low, ci_high = (
            np.percentile(metrics_boot[m], 2.5) if metrics_boot[m] else np.nan,
            np.percentile(metrics_boot[m], 97.5) if metrics_boot[m] else np.nan
        )
        # Format as value (CI_low–CI_high)
        res[f"{m}_ci"] = f"{res[m]:.3f} ({ci_low:.3f}–{ci_high:.3f})" if not np.isnan(res[m]) else np.nan

    return res

def evaluate_by_group_with_ci(df, group_col, threshold=0.5, n_boot=1000):
    rows = []
    for val, sub in df.groupby(group_col):
        m = compute_metrics_with_ci(sub[y_true_col], sub[y_score_col], threshold, n_boot)
        m["group_var"] = group_col
        m["group"] = val
        rows.append(m)
    return pd.DataFrame(rows)

# ============================================================
# PCA + PredictedRisk
# ============================================================

numeric_features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

X_full_numeric = mendeley[numeric_features].copy()
X_full_pca = pca.transform(X_full_numeric)
y_score_col = "PredictedRisk"
mendeley[y_score_col] = calibrated_stacking.predict_proba(X_full_pca)[:, 1]

# ============================================================
# Subgroup creation
# ============================================================

mendeley["AgeGroup"] = pd.cut(mendeley["Age"], bins=[-np.inf,40,60,np.inf], labels=["<40","40–60",">60"])
mendeley["BP_Group"] = pd.cut(mendeley["RestingBP"], bins=[-np.inf,120,139,np.inf], labels=["Normal","Elevated","Hypertension"])
mendeley["Chol_Group"] = pd.cut(mendeley["Cholesterol"], bins=[-np.inf,200,239,np.inf], labels=["<200","200–239","≥240"])
mendeley["MaxHR_Group"] = pd.cut(mendeley["MaxHR"], bins=[-np.inf,120,160,np.inf], labels=["<120","120–160",">160"])
mendeley["Oldpeak_Group"] = pd.cut(mendeley["Oldpeak"], bins=[-np.inf,1,np.inf], labels=["0–1",">1"])

subgroup_vars = [
    "Sex","AgeGroup","ChestPainType","BP_Group","Chol_Group",
    "FastingBS","RestingECG","MaxHR_Group","ExerciseAngina",
    "Oldpeak_Group","ST_Slope"
]

# ============================================================
# Compute overall metrics
# ============================================================

overall_metrics = compute_metrics_with_ci(mendeley[y_true_col], mendeley[y_score_col], threshold, n_boot)
overall_df = pd.DataFrame([overall_metrics])
overall_df.insert(0,"group_var","overall")
overall_df.insert(1,"group","overall")

# ============================================================
# Compute subgroup metrics
# ============================================================

all_results = []
for var in subgroup_vars:
    res = evaluate_by_group_with_ci(mendeley, var, threshold, n_boot)
    all_results.append(res)

subgroup_results = pd.concat(all_results, ignore_index=True)

# ============================================================
# Save CSVs
# ============================================================

overall_df.to_csv("Overall_Performance_with_CI_Formatted.csv", index=False)
subgroup_results.to_csv("ALL_Subgroup_Performance_with_CI_Formatted.csv", index=False)

print("✅ Generated formatted CSV files for paper:")
print(" - Overall_Performance_with_CI_Formatted.csv")
print(" - ALL_Subgroup_Performance_with_CI_Formatted.csv")

```

### Cell 39 — code (execution_count=36)
```python
# ============================================================
# Subgroup / Fairness Analysis with Bootstrap CIs (Paper-Ready)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

y_true_col = "HeartDisease"
threshold = 0.5
n_boot = 1000  # bootstrap iterations

# -----------------------------
# Helper functions
# -----------------------------

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1 - p))

def safe_div(n, d):
    return np.nan if d == 0 else n / d

def compute_metrics_with_ci(y_true, y_score, threshold=0.5, n_boot=1000):
    """
    Computes AUROC, calibration, threshold metrics, and bootstrap CIs.
    Formats sensitivity, specificity, PPV, NPV, and AUROC as value (CI_low–CI_high)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    res = {}

    # AUROC
    try:
        auc_val = roc_auc_score(y_true, y_score)
    except:
        auc_val = np.nan

    # Calibration
    try:
        X_logit = logit(y_score).reshape(-1,1)
        lr = LogisticRegression()
        lr.fit(X_logit, y_true)
        res["calib_intercept"] = lr.intercept_[0]
        res["calib_slope"] = lr.coef_[0][0]
    except:
        res["calib_intercept"] = np.nan
        res["calib_slope"] = np.nan

    # Threshold metrics
    y_pred = (y_score >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except:
        tn = fp = fn = tp = np.nan

    metrics = ["sensitivity", "specificity", "ppv", "npv"]
    values = [safe_div(tp, tp+fn), safe_div(tn, tn+fp), safe_div(tp, tp+fp), safe_div(tn, tn+fn)]

    for m, v in zip(metrics, values):
        res[m] = v

    # Sample info
    res["n"] = len(y_true)
    res["events"] = int(y_true.sum())
    res["event_rate"] = safe_div(res["events"], res["n"])
    res["threshold"] = threshold
    res["auc"] = auc_val

    # -----------------------------
    # Bootstrap CI
    # -----------------------------
    rng = np.random.RandomState(42)
    metrics_boot = {m: [] for m in metrics + ["auc"]}

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_score[idx]
        # Threshold metrics
        yb_pred = (yp >= threshold).astype(int)
        try:
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yb_pred, labels=[0,1]).ravel()
        except:
            continue
        metrics_boot["sensitivity"].append(safe_div(tp_b, tp_b+fn_b))
        metrics_boot["specificity"].append(safe_div(tn_b, tn_b+fp_b))
        metrics_boot["ppv"].append(safe_div(tp_b, tp_b+fp_b))
        metrics_boot["npv"].append(safe_div(tn_b, tn_b+fn_b))
        # AUROC
        try:
            metrics_boot["auc"].append(roc_auc_score(yt, yp))
        except:
            continue

    # Compute 95% CIs and format as value (CI_low–CI_high)
    for m in metrics + ["auc"]:
        if metrics_boot[m]:
            ci_low, ci_high = np.percentile(metrics_boot[m], [2.5, 97.5])
            res[f"{m}_ci"] = f"{res[m]:.3f} ({ci_low:.3f}–{ci_high:.3f})"
        else:
            res[f"{m}_ci"] = np.nan

    return res

def evaluate_by_group_with_ci(df, group_col, threshold=0.5, n_boot=1000):
    rows = []
    for val, sub in df.groupby(group_col):
        m = compute_metrics_with_ci(sub[y_true_col], sub[y_score_col], threshold, n_boot)
        m["group_var"] = group_col
        m["group"] = val
        rows.append(m)
    return pd.DataFrame(rows)

# ============================================================
# Step 1: PCA transform numeric features
# ============================================================

numeric_features = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

X_full_numeric = mendeley[numeric_features].copy()
X_full_pca = pca.transform(X_full_numeric)

# ============================================================
# Step 2: Compute predicted probabilities
# ============================================================

y_score_col = "PredictedRisk"
mendeley[y_score_col] = calibrated_stacking.predict_proba(X_full_pca)[:, 1]

# ============================================================
# Step 3: Create subgroup variables
# ============================================================

mendeley["AgeGroup"] = pd.cut(mendeley["Age"], bins=[-np.inf,40,60,np.inf], labels=["<40","40–60",">60"])
mendeley["BP_Group"] = pd.cut(mendeley["RestingBP"], bins=[-np.inf,120,139,np.inf], labels=["Normal","Elevated","Hypertension"])
mendeley["Chol_Group"] = pd.cut(mendeley["Cholesterol"], bins=[-np.inf,200,239,np.inf], labels=["<200","200–239","≥240"])
mendeley["MaxHR_Group"] = pd.cut(mendeley["MaxHR"], bins=[-np.inf,120,160,np.inf], labels=["<120","120–160",">160"])
mendeley["Oldpeak_Group"] = pd.cut(mendeley["Oldpeak"], bins=[-np.inf,1,np.inf], labels=["0–1",">1"])

subgroup_vars = [
    "Sex","AgeGroup","ChestPainType","BP_Group","Chol_Group",
    "FastingBS","RestingECG","MaxHR_Group","ExerciseAngina",
    "Oldpeak_Group","ST_Slope"
]

# ============================================================
# Step 4: Compute overall metrics
# ============================================================

overall_metrics = compute_metrics_with_ci(mendeley[y_true_col], mendeley[y_score_col], threshold, n_boot)
overall_df = pd.DataFrame([overall_metrics])
overall_df.insert(0,"group_var","overall")
overall_df.insert(1,"group","overall")

# ============================================================
# Step 5: Compute subgroup metrics
# ============================================================

all_results = []
for var in subgroup_vars:
    res = evaluate_by_group_with_ci(mendeley, var, threshold, n_boot)
    all_results.append(res)

subgroup_results = pd.concat(all_results, ignore_index=True)

# ============================================================
# Step 6: Save outputs
# ============================================================

overall_df.to_csv("Overall_Performance_with_CI_Formatted.csv", index=False)
subgroup_results.to_csv("ALL_Subgroup_Performance_with_CI_Formatted.csv", index=False)

print("✅ Generated formatted CSV files for paper (AUROC + threshold metrics with CIs):")
print(" - Overall_Performance_with_CI_Formatted.csv")
print(" - ALL_Subgroup_Performance_with_CI_Formatted.csv")

from google.colab import files

files.download("Overall_Performance_with_CI_Formatted.csv")
files.download("ALL_Subgroup_Performance_with_CI_Formatted.csv")

```

### Cell 40 — code (execution_count=None)
```python
# ============================================================
# Forest Plots for Subgroup Metrics (AUROC + Threshold Metrics)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load subgroup metrics CSV
subgroup_results = pd.read_csv("ALL_Subgroup_Performance_with_CI_Formatted.csv")

# Metrics to plot
metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

# Function to extract value and CI from formatted string "value (CI_low–CI_high)"
def parse_metric_ci(s):
    try:
        val, ci = s.split(" (")
        ci_low, ci_high = ci[:-1].split("–")
        return float(val), float(ci_low), float(ci_high)
    except:
        return np.nan, np.nan, np.nan

# Loop over metrics and generate forest plots
for metric in metrics:
    values, ci_lows, ci_highs = [], [], []
    labels = []

    for idx, row in subgroup_results.iterrows():
        val_str = row[f"{metric}_ci"]
        val, ci_low, ci_high = parse_metric_ci(val_str)
        values.append(val)
        ci_lows.append(val - ci_low)
        ci_highs.append(ci_high - val)
        labels.append(f"{row['group_var']}={row['group']}")

    # Plot
    plt.figure(figsize=(8, max(4, len(labels)*0.3)))
    plt.errorbar(values, range(len(labels)), xerr=[ci_lows, ci_highs],
                 fmt='o', color='blue', ecolor='lightgray', elinewidth=3, capsize=4)
    plt.yticks(range(len(labels)), labels)
    plt.axvline(0.5 if metric != "auc" else 0.5, color='red', linestyle='--', label='Reference')
    plt.xlabel(metric.upper())
    plt.title(f"Forest Plot of {metric.upper()} Across Subgroups")
    plt.gca().invert_yaxis()  # Highest on top
    plt.tight_layout()
    plt.show()

```

### Cell 41 — code (execution_count=None)
```python

```

### Cell 42 — markdown (execution_count=None)
```text
#**Reviewer 1, Concern 11** For OVEC
```

### Cell 43 — markdown (execution_count=None)
```text
⭐ SECTION 1 — IMPORTS
```

### Cell 44 — code (execution_count=None)
```python
# ============================================================
# CLEAN HEART DISEASE MODEL + SUBGROUP FAIRNESS PIPELINE
# Fully Fixed Version (Voting Classifier)
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import matplotlib.pyplot as plt
```

### Cell 45 — markdown (execution_count=None)
```text
⭐ SECTION 2 — LOAD + CLEAN DATA
```

### Cell 46 — code (execution_count=None)
```python
# ============================================================
# 1. LOAD & CLEAN mendeley
# ============================================================

# If your mendeleyset is already in "mendeley", skip next line
# mendeley = pd.read_csv("/content/drive/MyDrive/heart_mendeleyset.csv")

# Clean duplicates & missing target
mendeley = mendeley.drop_duplicates()
mendeley = mendeley.dropna(subset=["HeartDisease"])

# Ensure numeric target
mendeley["HeartDisease"] = mendeley["HeartDisease"].astype(int)

# Remove any previous subgroup columns (critical)
group_cols = [
    "AgeGroup","BP_Group","Chol_Group","MaxHR_Group","Oldpeak_Group"
]
for col in group_cols:
    if col in mendeley.columns:
        mendeley = mendeley.drop(columns=[col])

```

### Cell 47 — markdown (execution_count=None)
```text
⭐ SECTION 3 — PREPROCESSING + VOTING CLASSIFIER
```

### Cell 48 — code (execution_count=None)
```python
# ============================================================
# 2. FEATURES & PREPROCESSING
# ============================================================

X = mendeley.drop("HeartDisease", axis=1)
y = mendeley["HeartDisease"]

# Categorical features
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# Numeric features
num_cols = [c for c in X.columns if c not in cat_cols]

# Preprocessor: scale numeric + one-hot encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)
```

### Cell 49 — markdown (execution_count=None)
```text
⭐ SECTION 4 — StakING CLASSIFIER MODEL
```

### Cell 50 — code (execution_count=None)
```python
# ============================================================
# 3. VOTING CLASSIFIER (Your Choice: Option C)
# ============================================================
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


# FULL PIPELINE: preprocess → SMOTE → voting classifier
clf_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", calibrated_stacking)
])

```

### Cell 51 — markdown (execution_count=None)
```text
⭐ SECTION 5 — TRAIN / TEST SPLIT + TRAINING
```

### Cell 52 — code (execution_count=None)
```python
# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# TRAIN MODEL
clf_pipeline.fit(X_train, y_train)

# Predict probabilities (PredictedRisk)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# Attach test predictions
test_df = X_test.copy()
test_df["HeartDisease"] = y_test.values
test_df["PredictedRisk"] = y_proba

```

### Cell 53 — markdown (execution_count=None)
```text
⭐ SECTION 6 — EVALUATION ON TEST SET
```

### Cell 54 — code (execution_count=None)
```python
# ============================================================
# 5. OVERALL TEST PERFORMANCE
# ============================================================

auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
spec = tn / (tn + fp)
prec = precision_score(y_test, y_pred)
npv = tn / (tn + fn)

print("=== Test-set Performance ===")
print(f"AUROC:       {auc:.3f}")
print(f"Accuracy:    {acc:.3f}")
print(f"Sensitivity: {sens:.3f}")
print(f"Specificity: {spec:.3f}")
print(f"PPV:         {prec:.3f}")
print(f"NPV:         {npv:.3f}")

```

### Cell 55 — markdown (execution_count=None)
```text
⭐ SECTION 7 — CREATE SUBGROUP VARIABLES (TEST SET ONLY)
```

### Cell 56 — code (execution_count=None)
```python
# ============================================================
# 6. SUBGROUP CREATION (TEST SET ONLY)
# ============================================================

fair = test_df.copy()

fair["AgeGroup"] = pd.cut(fair["Age"], [-np.inf,40,60,np.inf], labels=["<40","40–60",">60"])
fair["BP_Group"] = pd.cut(fair["RestingBP"], [-np.inf,120,139,np.inf], labels=["Normal","Elevated","Hypertension"])
fair["Chol_Group"] = pd.cut(fair["Cholesterol"], [-np.inf,200,239,np.inf], labels=["<200","200–239","≥240"])
fair["MaxHR_Group"] = pd.cut(fair["MaxHR"], [-np.inf,120,160,np.inf], labels=["<120","120–160",">160"])
fair["Oldpeak_Group"] = pd.cut(fair["Oldpeak"], [-np.inf,1,np.inf], labels=["0–1",">1"])

subgroup_vars = [
    "Sex","AgeGroup","ChestPainType","BP_Group","Chol_Group",
    "FastingBS","RestingECG","MaxHR_Group","ExerciseAngina",
    "Oldpeak_Group","ST_Slope"
]

```

### Cell 57 — markdown (execution_count=None)
```text
⭐ SECTION 8 — BOOTSTRAP METRICS + FAIRNESS ANALYSIS
```

### Cell 58 — code (execution_count=None)
```python
# ============================================================
# 7. FAIRNESS ANALYSIS (BOOTSTRAP CIs)
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

threshold = 0.5
n_boot = 1000
y_true_col = "HeartDisease"
y_score_col = "PredictedRisk"

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def safe_div(a,b): return np.nan if b==0 else a/b

def compute_metrics_with_ci(y_true, y_score, threshold=0.5, n_boot=1000):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {}

    # AUROC
    try:
        auc_val = roc_auc_score(y_true, y_score)
    except:
        auc_val = np.nan

    # Calibration
    try:
        lr = LogisticRegression().fit(logit(y_score).reshape(-1,1), y_true)
        out["calib_intercept"] = lr.intercept_[0]
        out["calib_slope"] = lr.coef_[0][0]
    except:
        out["calib_intercept"] = np.nan
        out["calib_slope"] = np.nan

    # Threshold metrics
    y_pred = (y_score >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except:
        tn=fp=fn=tp=np.nan

    out["sensitivity"] = safe_div(tp, tp+fn)
    out["specificity"] = safe_div(tn, tn+fp)
    out["ppv"] = safe_div(tp, tp+fp)
    out["npv"] = safe_div(tn, tn+fn)
    out["auc"] = auc_val
    out["n"] = len(y_true)
    out["events"] = y_true.sum()

    # Bootstrap CIs
    rng = np.random.RandomState(42)
    metrics_list = ["sensitivity","specificity","ppv","npv","auc"]
    boot = {m:[] for m in metrics_list}

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt = y_true[idx]
        yp = y_score[idx]
        yp_bin = (yp>=threshold)

        try:
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yp_bin, labels=[0,1]).ravel()
        except:
            continue

        boot["sensitivity"].append(safe_div(tp_b, tp_b+fn_b))
        boot["specificity"].append(safe_div(tn_b, tn_b+fp_b))
        boot["ppv"].append(safe_div(tp_b, tp_b+fp_b))
        boot["npv"].append(safe_div(tn_b, tn_b+fn_b))

        try:
            boot["auc"].append(roc_auc_score(yt, yp))
        except:
            pass

    for m in metrics_list:
        if len(boot[m])>0:
            low, high = np.percentile(boot[m],[2.5,97.5])
            out[f"{m}_ci"] = f"{out[m]:.3f} ({low:.3f}–{high:.3f})"
        else:
            out[f"{m}_ci"] = np.nan

    return out

def evaluate_by_group(df, group_col):
    rows = []
    for val, subset in df.groupby(group_col):
        m = compute_metrics_with_ci(
            subset[y_true_col],
            subset[y_score_col],
            threshold,
            n_boot
        )
        m["group_var"] = group_col
        m["group"] = val
        rows.append(m)
    return pd.DataFrame(rows)

# Overall
overall_df = evaluate_by_group(fair.assign(overall="all"), "overall")

# Subgroups
subgroup_frames = []
for var in subgroup_vars:
    subgroup_frames.append(evaluate_by_group(fair, var))

subgroup_df = pd.concat(subgroup_frames, ignore_index=True)

# Save
overall_df.to_csv("Overall_Performance_with_CI.csv", index=False)
subgroup_df.to_csv("All_Subgroup_Performance_with_CI.csv", index=False)

print("DONE → CSVs generated")

```

### Cell 59 — code (execution_count=None)
```python
import pandas as pd
from openpyxl import Workbook
import os

# Load CSVs (must be in your working directory)
overall_df = pd.read_csv("Overall_Performance_with_CI.csv")
subgroup_df = pd.read_csv("All_Subgroup_Performance_with_CI.csv")

# Save in current folder
output_path = "Subgroup_Fairness_Results.xlsx"

wb = Workbook()

# Sheet 1 - Overall
ws1 = wb.active
ws1.title = "Overall Performance"
ws1.append(list(overall_df.columns))
for row in overall_df.itertuples(index=False):
    ws1.append(list(row))

# Sheet 2 - Subgroups
ws2 = wb.create_sheet("Subgroup Performance")
ws2.append(list(subgroup_df.columns))
for row in subgroup_df.itertuples(index=False):
    ws2.append(list(row))

# Save file
wb.save(output_path)

print("Excel file saved as:", output_path)
print("Absolute path:", os.path.abspath(output_path))

```

### Cell 60 — code (execution_count=None)
```python
from google.colab import files
files.download("Subgroup_Fairness_Results.xlsx")
```

### Cell 61 — markdown (execution_count=None)
```text
⭐ SECTION 9 — FOREST PLOTS
```

### Cell 62 — code (execution_count=None)
```python
# ============================================================
# 8. FOREST PLOTS (AUROC + Sens/Spec/PPV/NPV)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

df = subgroup_df.copy()

def parse_ci(s):
    try:
        val, rest = s.split(" (")
        low, high = rest[:-1].split("–")
        return float(val), float(low), float(high)
    except:
        return np.nan, np.nan, np.nan

metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

for metric in metrics:
    vals, lows, highs, labels = [], [], [], []

    for _, row in df.iterrows():
        v, lo, hi = parse_ci(row[f"{metric}_ci"])
        vals.append(v)
        lows.append(v - lo)
        highs.append(hi - v)
        labels.append(f"{row['group_var']}={row['group']}")

    plt.figure(figsize=(8, max(5, len(labels)*0.35)))
    plt.errorbar(vals, range(len(vals)),
                 xerr=[lows, highs], fmt='o',
                 ecolor="gray", capsize=4)
    plt.yticks(range(len(vals)), labels)
    plt.xlabel(metric.upper())
    plt.title(f"Forest Plot — {metric.upper()}")
    plt.axvline(0.5 if metric=="auc" else 0.5, color="red", ls="--")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

```

### Cell 63 — code (execution_count=None)
```python
# ============================================================
# 8. NATURE-STYLE FOREST PLOTS (AUROC + Sens/Spec/PPV/NPV)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

df = subgroup_df.copy()

def parse_ci(s):
    """Extract value, lower, and upper from 'value (low–high)' format."""
    try:
        val, rest = s.split(" (")
        low, high = rest[:-1].split("–")
        return float(val), float(low), float(high)
    except:
        return np.nan, np.nan, np.nan


metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

for metric in metrics:
    vals, ci_low, ci_high, labels = [], [], [], []

    for _, row in df.iterrows():
        v, low, high = parse_ci(row[f"{metric}_ci"])
        vals.append(v)
        ci_low.append(low)
        ci_high.append(high)
        labels.append(f"{row['group_var']} = {row['group']}")

    vals = np.array(vals)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    # Convert to CI widths
    lower_err = vals - ci_low
    upper_err = ci_high - vals

    y_pos = np.arange(len(vals))

    plt.figure(figsize=(9, max(6, len(vals)*0.33)))

    # ---- CI SHADING (Nature style) ----
    plt.hlines(y_pos, ci_low, ci_high, color='black', linewidth=1.3)
    plt.fill_betweenx(y_pos, ci_low, ci_high, color='skyblue', alpha=0.25)

    # ---- Point Estimates ----
    plt.scatter(vals, y_pos, color='blue', s=50, zorder=3)

    # ---- Reference Line ----
    ref = 0.5
    plt.axvline(ref, color='red', linestyle='--', linewidth=1.2)

    # ---- Labels ----
    plt.yticks(y_pos, labels, fontsize=9)
    plt.xlabel(metric.upper(), fontsize=12)
    plt.title(f"Forest Plot — {metric.upper()}", fontsize=14)

    plt.gca().invert_yaxis()  # Put top row first
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

```

### Cell 64 — code (execution_count=None)
```python
# ============================================================
# 8. NATURE-STYLE FOREST PLOTS WITH SAMPLE SIZES
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

df = subgroup_df.copy()

def parse_ci(s):
    """Extract value, lower, and upper from 'value (low–high)' string."""
    try:
        val, rest = s.split(" (")
        low, high = rest[:-1].split("–")
        return float(val), float(low), float(high)
    except:
        return np.nan, np.nan, np.nan


metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

for metric in metrics:
    vals, ci_low, ci_high, labels = [], [], [], []

    for _, row in df.iterrows():
        v, low, high = parse_ci(row[f"{metric}_ci"])
        vals.append(v)
        ci_low.append(low)
        ci_high.append(high)

        # ------------------------------
        # Add sample size next to label
        # ------------------------------
        label_n = int(row["n"])
        label = f"{row['group_var']}={row['group']} (n={label_n})"
        labels.append(label)

    vals = np.array(vals)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    # Convert to CI distances
    lower_err = vals - ci_low
    upper_err = ci_high - vals

    y_pos = np.arange(len(vals))

    plt.figure(figsize=(10, max(6, len(vals)*0.33)))

    # ---- CI Shading ----
    plt.hlines(y_pos, ci_low, ci_high, color='black', linewidth=1.3)
    plt.fill_betweenx(y_pos, ci_low, ci_high, color='skyblue', alpha=0.25)

    # ---- Point Estimates ----
    plt.scatter(vals, y_pos, color='blue', s=50, zorder=3)

    # ---- Reference Line ----
    plt.axvline(0.5, color='red', linestyle='--', linewidth=1.2)

    # ---- Formatting ----
    plt.yticks(y_pos, labels, fontsize=9)
    plt.xlabel(metric.upper(), fontsize=12)
    plt.title(f"Forest Plot — {metric.upper()}", fontsize=14)

    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

```

### Cell 65 — code (execution_count=None)
```python
# ============================================================
# 8. NATURE-STYLE FOREST PLOTS WITH HIGHLIGHTED CIs
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

df = subgroup_df.copy()

def parse_ci(s):
    """Extract value, lower, and upper from 'value (low–high)' string."""
    try:
        val, rest = s.split(" (")
        low, high = rest[:-1].split("–")
        return float(val), float(low), float(high)
    except:
        return np.nan, np.nan, np.nan


metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

for metric in metrics:
    vals, ci_low_vals, ci_high_vals, labels = [], [], [], []

    for _, row in df.iterrows():
        v, lo, hi = parse_ci(row[f"{metric}_ci"])
        vals.append(v)
        ci_low_vals.append(lo)
        ci_high_vals.append(hi)

        # Add sample size
        n = int(row["n"])
        label = f"{row['group_var']}={row['group']} (n={n})"
        labels.append(label)

    vals = np.array(vals)
    ci_low_vals = np.array(ci_low_vals)
    ci_high_vals = np.array(ci_high_vals)

    y_pos = np.arange(len(vals))

    plt.figure(figsize=(10, max(6, len(vals)*0.33)))

    # ============================================================
    # DRAW CI + POINT + HIGHLIGHT
    # ============================================================
    for i in range(len(vals)):
        v = vals[i]
        lo = ci_low_vals[i]
        hi = ci_high_vals[i]

        # Highlight if CI crosses 0.5
        crosses = (lo < 0.5) and (hi > 0.5)

        if crosses:
            ci_color = "lightcoral"
            point_color = "darkred"
            text_color = "darkred"
        else:
            ci_color = "skyblue"
            point_color = "blue"
            text_color = "black"

        # CI line + shading
        plt.hlines(y_pos[i], lo, hi, color='black', linewidth=1.3)
        plt.fill_betweenx([y_pos[i]], lo, hi, color=ci_color, alpha=0.30)

        # Point estimate
        plt.scatter(v, y_pos[i], color=point_color, s=55, zorder=3)

        # Label
        plt.text(0.02, y_pos[i], labels[i], fontsize=9, color=text_color)

    # Reference line at 0.5
    plt.axvline(0.5, color='red', linestyle='--', linewidth=1.2)

    plt.yticks(y_pos, [""]*len(vals))  # we draw labels manually
    plt.xlabel(metric.upper(), fontsize=12)
    plt.title(f"Forest Plot — {metric.upper()}", fontsize=14)

    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

```

### Cell 66 — code (execution_count=None)
```python
# ============================================================
# 8. NATURE-STYLE FOREST PLOTS WITH HIGHLIGHT + NON-OVERLAPPING LABELS
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

df = subgroup_df.copy()

def parse_ci(s):
    try:
        val, rest = s.split(" (")
        low, high = rest[:-1].split("–")
        return float(val), float(low), float(high)
    except:
        return np.nan, np.nan, np.nan


metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

for metric in metrics:
    vals, ci_low_vals, ci_high_vals, labels = [], [], [], []

    for _, row in df.iterrows():
        v, lo, hi = parse_ci(row[f"{metric}_ci"])
        vals.append(v)
        ci_low_vals.append(lo)
        ci_high_vals.append(hi)

        # Sample size
        n = int(row["n"])
        label = f"{row['group_var']}={row['group']} (n={n})"
        labels.append(label)

    vals = np.array(vals)
    ci_low_vals = np.array(ci_low_vals)
    ci_high_vals = np.array(ci_high_vals)
    y_pos = np.arange(len(vals))

    # -----------------------------
    # FIGURE + EXPANDED LEFT MARGIN
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, max(8, len(vals)*0.35)))
    plt.subplots_adjust(left=0.32)  # <<< Key fix: larger left space

    # -----------------------------
    # PLOT EACH SUBGROUP
    # -----------------------------
    for i in range(len(vals)):
        v = vals[i]
        lo = ci_low_vals[i]
        hi = ci_high_vals[i]

        # Highlight where CI crosses 0.5
        crosses = (lo < 0.5) and (hi > 0.5)
        if crosses:
            ci_color = "lightcoral"
            point_color = "darkred"
            text_color = "darkred"
        else:
            ci_color = "skyblue"
            point_color = "blue"
            text_color = "black"

        # CI line + shaded area
        ax.hlines(y_pos[i], lo, hi, color='black', linewidth=1.2)
        ax.fill_betweenx([y_pos[i]], lo, hi, color=ci_color, alpha=0.33)

        # Point estimate
        ax.scatter(v, y_pos[i], color=point_color, s=60, zorder=3)

        # Draw label OUTSIDE plot
        ax.text(-0.02, y_pos[i], labels[i], fontsize=9, color=text_color,
                ha='right', va='center', transform=ax.get_yaxis_transform())

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f"Forest Plot — {metric.upper()}", fontsize=15)

    ax.set_yticks([])  # remove default tick labels
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

```

### Cell 67 — code (execution_count=None)
```python
# ============================================================
# 8. NATURE-STYLE FOREST PLOTS WITH HIGHLIGHT + LEGEND
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

df = subgroup_df.copy()

def parse_ci(s):
    try:
        val, rest = s.split(" (")
        low, high = rest[:-1].split("–")
        return float(val), float(low), float(high)
    except:
        return np.nan, np.nan, np.nan

metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

for metric in metrics:
    vals, ci_low_vals, ci_high_vals, labels = [], [], [], []

    for _, row in df.iterrows():
        v, lo, hi = parse_ci(row[f"{metric}_ci"])
        vals.append(v)
        ci_low_vals.append(lo)
        ci_high_vals.append(hi)

        n = int(row["n"])
        label = f"{row['group_var']}={row['group']} (n={n})"
        labels.append(label)

    vals = np.array(vals)
    ci_low_vals = np.array(ci_low_vals)
    ci_high_vals = np.array(ci_high_vals)
    y_pos = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(12, max(8, len(vals)*0.35)))
    plt.subplots_adjust(left=0.32)

    # -------------------------------------------
    # DRAW FOREST PLOT
    # -------------------------------------------
    for i in range(len(vals)):
        v = vals[i]
        lo = ci_low_vals[i]
        hi = ci_high_vals[i]

        # Highlight if CI crosses 0.5
        crosses = (lo < 0.5) and (hi > 0.5)

        if crosses:
            ci_color = "lightcoral"
            point_color = "darkred"
            text_color = "darkred"
        else:
            ci_color = "skyblue"
            point_color = "blue"
            text_color = "black"

        # CI line + shaded region
        ax.hlines(y_pos[i], lo, hi, color='black', linewidth=1.2)
        ax.fill_betweenx([y_pos[i]], lo, hi, color=ci_color, alpha=0.33)

        # Point estimate
        ax.scatter(v, y_pos[i], color=point_color, s=60, zorder=3)

        # Label outside plot
        ax.text(-0.02, y_pos[i], labels[i], fontsize=9, color=text_color,
                ha='right', va='center', transform=ax.get_yaxis_transform())

    # ----------------------------------------------------
    # LEGEND (Clean, Professional)
    # ----------------------------------------------------
    blue_patch = mpatches.Patch(color="skyblue", label="CI does NOT cross 0.5")
    red_patch = mpatches.Patch(color="lightcoral", label="CI crosses 0.5 (Fairness concern)")
    blue_dot = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', label='Point estimate (no issue)')
    red_dot  = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='darkred', label='Point estimate (highlighted)')

    ax.legend(
        handles=[blue_patch, red_patch, blue_dot, red_dot],
        fontsize=9,
        frameon=True,
        loc="upper left",
        borderpad=0.6
    )

    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.2)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f"Forest Plot — {metric.upper()}", fontsize=15)

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

```

### Cell 68 — code (execution_count=None)
```python
# ============================================================
# 8. NATURE-STYLE FOREST PLOTS WITH HIGHLIGHT + LEGEND
#    + AUTO SAVE & DOWNLOAD (PDF, 300 DPI)
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from google.colab import files
import os

# ------------------------------------------------------------
# COPY DATA
# ------------------------------------------------------------
df = subgroup_df.copy()

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------
output_dir = "forest_plots"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# CI PARSER
# ------------------------------------------------------------
def parse_ci(s):
    try:
        val, rest = s.split(" (")
        low, high = rest[:-1].split("–")
        return float(val), float(low), float(high)
    except:
        return np.nan, np.nan, np.nan

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
metrics = ["auc", "sensitivity", "specificity", "ppv", "npv"]

# ------------------------------------------------------------
# LOOP THROUGH METRICS
# ------------------------------------------------------------
for metric in metrics:
    vals, ci_low_vals, ci_high_vals, labels = [], [], [], []

    for _, row in df.iterrows():
        v, lo, hi = parse_ci(row[f"{metric}_ci"])
        vals.append(v)
        ci_low_vals.append(lo)
        ci_high_vals.append(hi)

        n = int(row["n"])
        label = f"{row['group_var']}={row['group']} (n={n})"
        labels.append(label)

    vals = np.array(vals)
    ci_low_vals = np.array(ci_low_vals)
    ci_high_vals = np.array(ci_high_vals)
    y_pos = np.arange(len(vals))

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, max(8, len(vals) * 0.35)))
    plt.subplots_adjust(left=0.32)

    # --------------------------------------------------------
    # DRAW FOREST PLOT
    # --------------------------------------------------------
    for i in range(len(vals)):
        v = vals[i]
        lo = ci_low_vals[i]
        hi = ci_high_vals[i]

        # Highlight if CI crosses 0.5
        crosses = (lo < 0.5) and (hi > 0.5)

        if crosses:
            ci_color = "lightcoral"
            point_color = "darkred"
            text_color = "darkred"
        else:
            ci_color = "skyblue"
            point_color = "blue"
            text_color = "black"

        # CI line
        ax.hlines(y_pos[i], lo, hi, color="black", linewidth=1.2)

        # CI shaded region
        ax.fill_betweenx(
            [y_pos[i]],
            lo,
            hi,
            color=ci_color,
            alpha=0.33
        )

        # Point estimate
        ax.scatter(
            v,
            y_pos[i],
            color=point_color,
            s=60,
            zorder=3
        )

        # Left-side labels (outside plot)
        ax.text(
            -0.02,
            y_pos[i],
            labels[i],
            fontsize=9,
            color=text_color,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform()
        )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------
    blue_patch = mpatches.Patch(
        color="skyblue",
        label="CI does NOT cross 0.5"
    )
    red_patch = mpatches.Patch(
        color="lightcoral",
        label="CI crosses 0.5 (Fairness concern)"
    )
    blue_dot = plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        label="Point estimate (no issue)"
    )
    red_dot = plt.Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="darkred",
        label="Point estimate (highlighted)"
    )

    ax.legend(
        handles=[blue_patch, red_patch, blue_dot, red_dot],
        fontsize=9,
        frameon=True,
        loc="upper left",
        borderpad=0.6
    )

    # --------------------------------------------------------
    # AXES & STYLE
    # --------------------------------------------------------
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel(metric.upper(), fontsize=12)
    #ax.set_title(f"Forest Plot — {metric.upper()}", fontsize=15)

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE + DOWNLOAD (PDF, 300 DPI)
    # --------------------------------------------------------
    filename = f"forest_plot_{metric}.pdf"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(
        filepath,
        format="pdf",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    files.download(filepath)

```