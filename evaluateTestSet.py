import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ================================
# Config (Frozen)
# ================================
THRESHOLD = 0.57

# ================================
# Load model and TEST set
# ================================
model = keras.models.load_model("tof_model.h5")

X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

# ================================
# Predict
# ================================
scores = model.predict(X_test)[:, 0]
y_pred = (scores >= THRESHOLD).astype(int)

# ================================
# Metrics
# ================================
acc  = accuracy_score(Y_test, y_pred)
prec = precision_score(Y_test, y_pred, zero_division=0)
rec  = recall_score(Y_test, y_pred, zero_division=0)
f1   = f1_score(Y_test, y_pred, zero_division=0)

print("\n=== TEST SET RESULTS ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# ================================
# Confusion Matrix
# ================================
cm = confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(4,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Invalid", "Valid"],
    yticklabels=["Invalid", "Valid"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()