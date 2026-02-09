import numpy as np
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ================================
# Load model and validation set
# ================================
model = keras.models.load_model("tof_model.h5")

X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")

scores = model.predict(X_val)[:, 0]

print("t\tPrecision\tRecall\tF1")

thresholds = []
precisions = []
recalls = []
f1s = []

best_f1 = 0
best_t = 0

# ================================
# Sweep thresholds
# ================================
for t in np.linspace(0.01, 0.99, 99):
    y_pred = (scores >= t).astype(int)

    p = precision_score(Y_val, y_pred, zero_division=0)
    r = recall_score(Y_val, y_pred, zero_division=0)
    f1 = f1_score(Y_val, y_pred, zero_division=0)

    thresholds.append(t)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

    print(f"{t:.2f}\t{p:.3f}\t\t{r:.3f}\t\t{f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nBest threshold (max F1):", best_t)
print("Best F1 (VAL):", best_f1)

# ================================
# Plot metrics vs threshold
# ================================
plt.figure(figsize=(7,5))

plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1s, label="F1-score", linewidth=2)

# chosen threshold (center of plateau, not necessarily best_t)
CHOSEN_THRESHOLD = 0.57
plt.axvline(
    CHOSEN_THRESHOLD,
    color="k",
    linestyle="--",
    label=f"Chosen threshold = {CHOSEN_THRESHOLD}"
)

plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.title("Precision, Recall and F1 vs Threshold (Validation Set)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()