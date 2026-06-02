import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

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

# LOAD DATASET
df = pd.read_csv("../dataset/heart.csv")

# LABEL ENCODING
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

# FEATURES & TARGET
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL
model = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=300,
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# BASIC METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

# ADDITIONAL METRICS
specificity = tn / (tn + fp)
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)

# PRINT RESULTS
print("\n========== MLP RESULTS ==========\n")

print(f"Accuracy      : {accuracy:.2f}")
print(f"Precision     : {precision:.2f}")
print(f"Recall        : {recall:.2f}")
print(f"F1 Score      : {f1:.2f}")

print("\n========== ADDITIONAL METRICS ==========\n")

print(f"Specificity   : {specificity:.2f}")
print(f"AUC Score     : {auc:.2f}")
print(f"Brier Score   : {brier:.2f}")
print(f"MCC Score     : {mcc:.2f}")

print("\nCONFUSION MATRIX:\n")
print(cm)

print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))
