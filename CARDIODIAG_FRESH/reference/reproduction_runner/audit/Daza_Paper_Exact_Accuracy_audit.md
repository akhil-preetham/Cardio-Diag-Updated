# Static notebook audit

- Notebook: `C:\CARDIO_DIAG\Cardio-Diag-Approach-Code-main\CARDIODIAG_FRESH\reference\extracted_author_archives\base_reproduction\Daza_Base_Paper_Reproduced\Daza_Paper_Exact_Accuracy.ipynb`
- SHA-256: `990f24622ca3aefe2f62e37f7d9e1097fe14b118a49ee398ff3ee67568bfbe4d`
- Cells: 20
- Saved outputs: 17

## Imports
- `google.colab`
- `imblearn.over_sampling`
- `matplotlib.pyplot`
- `numpy`
- `os`
- `pandas`
- `random`
- `sklearn.calibration`
- `sklearn.ensemble`
- `sklearn.linear_model`
- `sklearn.metrics`
- `sklearn.model_selection`
- `sklearn.naive_bayes`
- `sklearn.neighbors`
- `sklearn.preprocessing`
- `sklearn.svm`
- `sklearn.tree`
- `xgboost`

## Method keywords
- CalibratedClassifierCV
- GridSearchCV
- SMOTE
- StackingClassifier
- train_test_split

## Risk flags
- **credential_or_mount**: cells [2]

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
```

### Cell 2 — code (execution_count=None)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 3 — markdown (execution_count=None)
```text
# **Base paper dataset and model**
```

### Cell 4 — code (execution_count=None)
```python
df = pd.read_csv('/content/drive/MyDrive/heart5.csv')

#first 5 values of data
df.head()

```

### Cell 5 — code (execution_count=None)
```python
# Preprocessing
df = df.dropna()  # Drop rows with missing values

# Convert categorical columns into binary (0 and 1)
# Assuming the dataset has columns 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'

# Convert 'Sex' to 0 and 1 (e.g., Male = 1, Female = 0)
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

# Encode 'ChestPainType' (use LabelEncoder for multi-class categorical variables)
chest_pain_encoder = LabelEncoder()
df['ChestPainType'] = chest_pain_encoder.fit_transform(df['ChestPainType'])

# Encode 'RestingECG'
resting_ecg_encoder = LabelEncoder()
df['RestingECG'] = resting_ecg_encoder.fit_transform(df['RestingECG'])

# Convert 'ExerciseAngina' to binary (e.g., Y = 1, N = 0)
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
df['ST_Slope'] = df['ST_Slope'].map({'Up': 1, 'Flat': 0, 'Down': 0})
```

### Cell 6 — code (execution_count=None)
```python
df['RestingECG']
```

### Cell 7 — code (execution_count=None)
```python
X = df.drop('HeartDisease', axis=1)  # Assuming 'target' is the label column
y = df['HeartDisease']
```

### Cell 8 — code (execution_count=None)
```python
print(df)
```

### Cell 9 — code (execution_count=None)
```python
# Apply oversampling using SMOTE to balance the dataset
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Cell 10 — code (execution_count=None)
```python
# Count class instances before balancing
print("Class distribution before balancing:")
print(y.value_counts())

# After SMOTE, count class instances
print("Class distribution after balancing:")
print(pd.Series(y_resampled).value_counts())
```

### Cell 11 — code (execution_count=None)
```python
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

```

### Cell 12 — markdown (execution_count=None)
```text
**Algorithms hyper parameter-tuning**
```

### Cell 13 — code (execution_count=None)
```python
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

#  C KNN
param_grid_knn = {
    'n_neighbors': [5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric':  ['minkowskii', 'euclidean', 'manhattan'],
}

# D Random Forest
param_grid_rf = {
    'n_estimators': [25, 50, 100],
    'max_depth': [5, None],
    'max_features': ['sqrt', 'log2'],
    'max_leaf_nodes': [5, 10, None],
    'criterion': ['gini', 'entropy']
}

# E Logistic Regression
param_grid_logistic = {
    'penalty': ['l1', 'l2', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']
}

# F SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear', 'poly', 'rbf', 'sigmoid']
}

# G Bayesian Networks
param_grid_bayes = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

# H AdaBoost
param_grid_ab = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.1, 1.0, 1.1, 1.2]
}

# I AdaBoost-SVM
param_grid_absvm = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.1, 1.0, 1.1, 1.2]
}

# J XGBoost Classifier
param_grid_xgb = {
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree' : [0.6, 0.8, 1.0]
}

# K XGBoost Regressor
param_grid_xgb_reg = {
    'nthread': [4],
    'objective': ['reg:linear'],
    'max_depth' : [5, 6, 7],
    'min_child_weight' : [4],
    'silent': [1],
    'subsample' : [0.7],
    'colsample_bytree' : [0.7]
}
```

### Cell 14 — markdown (execution_count=None)
```text
# **Algorithms optimization commands**
*   Perform grid search on the individual models
```

### Cell 15 — code (execution_count=None)
```python
# Perform grid search on the individual models

# A Decision Tree Grid Search (Base 1 Classifier)
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)
best_tree = grid_search_tree.best_estimator_
# Output the best parameters
print("Best Parameters of Decision Tree Grid Search (Base 1 Classifier):", best_tree)

# B Gradient Boosting Grid Search (Base 2 Classifier)
gradien = GradientBoostingClassifier()
grid_search_gradien = GridSearchCV(gradien, param_grid_gradien, cv=5, scoring='accuracy')
grid_search_gradien.fit(X_train, y_train)
best_gradien = grid_search_gradien.best_estimator_
# Output the best parameters
print("Best Parameters of Gradient Boosting Grid Search (Base 2 Classifier):", best_gradien)

# C K-Nearest Neighbors Grid Search
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_
print("Best Parameters of K-Nearest Neighbors Grid Search:", best_knn)

# D Random Forest Grid Search (Base 3 Classifier)
rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
# Output the best parameters
print("Best Parameters of Random Forest Grid Search (Base 3 Classifier):", best_rf)

# E Logistic Regression Grid Search (Meta Classifier)
logistic = LogisticRegression()
grid_search_logistic = GridSearchCV(logistic, param_grid_logistic, cv=5, scoring='accuracy')
grid_search_logistic.fit(X_train, y_train)
best_logistic = grid_search_logistic.best_estimator_
# Output the best parameters
print("Best Parameters of Logistic Regression Grid Search (Meta Classifier):", best_logistic)

'''# F SVM Grid Search
svm = SVC()
grid_search_svc = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)
best_svc = grid_search_svc.best_estimator_
print("Best Parameters of SVM Grid Search:", best_svc)
'''

# G Naive Bayes Grid Search (Base 4 Classifier)
bayes = GaussianNB()
grid_search_bayes = GridSearchCV(bayes, param_grid_bayes, cv=5, scoring='accuracy')
grid_search_bayes.fit(X_train, y_train)
best_bayes = grid_search_bayes.best_estimator_
# Output the best parameters
print("Best Parameters of Naive Bayes Grid Search (Base 4 Classifier):", best_bayes)

# H AdaBoost Grid Search
ab = AdaBoostClassifier()
grid_search_ab = GridSearchCV(ab, param_grid_ab, cv=5, scoring='accuracy')
grid_search_ab.fit(X_train, y_train)
best_ab = grid_search_ab.best_estimator_
print("Best Parameters of AdaBoost Grid Search):", best_ab)

# I AdaBoost SVM Grid Search
absvm = AdaBoostClassifier(SVC(probability=True)) # Changed: Added probability=True to SVC
grid_search_absvm = GridSearchCV(absvm, param_grid_absvm, cv=5, scoring='accuracy')
grid_search_absvm.fit(X_train, y_train)
best_absvm = grid_search_absvm.best_estimator_
# Output the best parameters
print("Best Parameters of AdaBoost SVM Grid Search:", best_absvm)

# J XGBoost Classifier Grid Search
xgb = XGBClassifier()
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)
best_xgb = grid_search_xgb.best_estimator_
print("Best Parameters of XGBoost Classifier Grid Search:", best_xgb)

# K Naive Bayes Grid Search
xgb_reg = XGBRegressor()
grid_search_xgb_reg = GridSearchCV(xgb_reg, param_grid_xgb_reg, cv=5, scoring='accuracy')
grid_search_xgb_reg.fit(X_train, y_train)
best_xgb_reg = grid_search_xgb_reg.best_estimator_
print("Best Parameters of K Naive Bayes Grid Search:", best_xgb_reg )
```

### Cell 16 — markdown (execution_count=None)
```text
**Proposed Base Paper Stacking Model**


*   Base-----1. Decision Tree, 2. Random Forest, 3. Gradient Boosting, 4. Bayesian Network
*   Meta-----Logistic Regression


```

### Cell 17 — code (execution_count=None)
```python
# ============================================================
# Full Reproducible Stacking Classifier Pipeline
# ============================================================

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV

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
best_rf.random_state = SEED
best_tree.random_state = SEED
best_gradien.random_state = SEED
best_bayes.random_state = SEED
best_logistic.random_state = SEED

# ============================================================
# 3️⃣ Create the Stacking Classifier
# ============================================================
stacking_classifier = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('tree', best_tree),
        ('gradien', best_gradien),
        ('bayes', best_bayes)
    ],
    final_estimator=best_logistic,
    cv=5
)

```

### Cell 18 — code (execution_count=None)
```python
# Calibrate the Stacking Classifier
calibrated_stacking = CalibratedClassifierCV(stacking_classifier, method='sigmoid', cv=5)
calibrated_stacking.fit(X_train, y_train)
```

### Cell 19 — code (execution_count=None)
```python
# Fit the model with early stopping
n_epochs = 100
best_accuracy = 0
patience = 5
stopping_counter = 0

for epoch in range(n_epochs):
    calibrated_stacking.fit(X_train, y_train)
    y_pred = calibrated_stacking.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        stopping_counter = 0  # Reset the counter if we improve
    else:
        stopping_counter += 1  # Increment counter if no improvement

    if stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
```

### Cell 20 — code (execution_count=None)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt

# Predict on the test data
y_pred = calibrated_stacking.predict(X_test)
y_pred_proba = calibrated_stacking.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Print evaluation results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'AUC-ROC: {auc:.4f}')

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

```