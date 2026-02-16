import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, f1_score

# ------------------------------------------------------
# 1. FETCH DATASET FROM UCI
# ------------------------------------------------------
online_shoppers = fetch_ucirepo(id=468)

X = online_shoppers.data.features
y = online_shoppers.data.targets.iloc[:, 0].astype(int)

# ------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# ------------------------------------------------------
# 3. TRAIN–TEST SPLIT
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

results = []

# ------------------------------------------------------
# 4. LOGISTIC REGRESSION (FIXED – SCALED)
# ------------------------------------------------------
log_model = Pipeline([
    ("prep", preprocess),
    ("scale", StandardScaler(with_mean=False)),
    ("model", LogisticRegression(max_iter=3000))
])

log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

results.append([
    "Logistic Regression",
    accuracy_score(y_test, y_pred_log),
    precision_score(y_test, y_pred_log),
    f1_score(y_test, y_pred_log)
])

# ------------------------------------------------------
# 5. LINEAR REGRESSION (THRESHOLDED)
# ------------------------------------------------------
lin_model = Pipeline([
    ("prep", preprocess),
    ("scale", StandardScaler(with_mean=False)),
    ("model", LinearRegression())
])

lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
y_pred_lin_class = (y_pred_lin >= 0.5).astype(int)

results.append([
    "Linear Regression",
    accuracy_score(y_test, y_pred_lin_class),
    precision_score(y_test, y_pred_lin_class),
    f1_score(y_test, y_pred_lin_class)
])

# ------------------------------------------------------
# 6. MULTIPLE LINEAR REGRESSION
# ------------------------------------------------------
results.append([
    "Multiple Linear Regression",
    accuracy_score(y_test, y_pred_lin_class),
    precision_score(y_test, y_pred_lin_class),
    f1_score(y_test, y_pred_lin_class)
])

# ------------------------------------------------------
# 7. DECISION TREE
# ------------------------------------------------------
dt_model = Pipeline([
    ("prep", preprocess),
    ("model", DecisionTreeClassifier(random_state=42))
])

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

results.append([
    "Decision Tree",
    accuracy_score(y_test, y_pred_dt),
    precision_score(y_test, y_pred_dt),
    f1_score(y_test, y_pred_dt)
])



# ------------------------------------------------------
# RESULTS TABLE
# ------------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "F1 Score"]
)

print("\n......MODEL COMPARISON......")
print(results_df)


import matplotlib.pyplot as plt
import numpy as np

# Extract values
models = results_df["Model"]
accuracy = results_df["Accuracy"]
precision = results_df["Precision"]
f1 = results_df["F1 Score"]

# X-axis positions
x = np.arange(len(models))
width = 0.25

# Create bar chart
plt.figure(figsize=(10, 6))

plt.bar(x - width, accuracy, width, label="Accuracy")
plt.bar(x, precision, width, label="Precision")
plt.bar(x + width, f1, width, label="F1 Score")

# Labels and title
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Performance Comparison of Machine Learning Models")
plt.xticks(x, models, rotation=30)
plt.legend()

plt.tight_layout()
plt.show()

