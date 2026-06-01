import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    brier_score_loss,
    matthews_corrcoef
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("../dataset/heart.csv")

# =========================
# DISPLAY FIRST 5 ROWS
# =========================

print("\nFIRST 5 ROWS OF DATASET:\n")
print(df.head())

# =========================
# CHECK NULL VALUES
# =========================

print("\nNULL VALUES:\n")
print(df.isnull().sum())

# =========================
# LABEL ENCODING
# =========================

le = LabelEncoder()

categorical_columns = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# =========================
# FEATURES & TARGET
# =========================

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# LOGISTIC REGRESSION MODEL
# =========================

model = LogisticRegression(max_iter=1000)

# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# Probability Predictions
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# BASIC METRICS
# =========================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

# =========================
# ADDITIONAL METRICS
# =========================

# Specificity
specificity = tn / (tn + fp)

# AUC Score
auc = roc_auc_score(y_test, y_prob)

# Brier Score
brier = brier_score_loss(y_test, y_prob)

# MCC Score
mcc = matthews_corrcoef(y_test, y_pred)

# =========================
# PRINT RESULTS
# =========================

print("\n========== LOGISTIC REGRESSION RESULTS ==========\n")

print(f"Accuracy      : {accuracy:.2f}")
print(f"Precision     : {precision:.2f}")
print(f"Recall        : {recall:.2f}")
print(f"F1 Score      : {f1:.2f}")

# =========================
# PRINT ADDITIONAL METRICS
# =========================

print("\n========== ADDITIONAL METRICS ==========\n")

print(f"Specificity   : {specificity:.2f}")
print(f"AUC Score     : {auc:.2f}")
print(f"Brier Score   : {brier:.2f}")
print(f"MCC Score     : {mcc:.2f}")

# =========================
# PRINT CONFUSION MATRIX
# =========================

print("\nCONFUSION MATRIX:\n")
print(cm)

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))