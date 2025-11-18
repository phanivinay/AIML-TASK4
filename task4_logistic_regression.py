# Task 4: Logistic Regression Classification
# -------------------------------
# Requirements:
# - Binary classification using Logistic Regression
# - Train/Test split
# - Standardization
# - Evaluation: Confusion Matrix, Precision, Recall, ROC-AUC
# - Threshold tuning
# -------------------------------

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt

# 1. Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Loaded Successfully!")
print("Shape:", X.shape)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression Model
model = LogisticRegression(max_iter=3000)
model.fit(X_train_scaled, y_train)

# 5. Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# -------------------------------
# Evaluation Metrics
# -------------------------------

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle="--")  # diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.show()

# -------------------------------
# 6. Threshold Tuning Example
# -------------------------------
custom_threshold = 0.3  # change threshold here
y_custom = (y_prob >= custom_threshold).astype(int)

cm_custom = confusion_matrix(y_test, y_custom)
print(f"\nConfusion Matrix at Threshold = {custom_threshold}")
print(cm_custom)

# -------------------------------
# 7. Print Sigmoid Example
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("\nSigmoid Example:")
print("Sigmoid(0) =", sigmoid(0))
print("Sigmoid(2) =", sigmoid(2))
print("Sigmoid(-2) =", sigmoid(-2))
