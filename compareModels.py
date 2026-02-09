import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ================================
# Config
# ================================
THRESHOLD = 0.57

# ================================
# Load test set
# ================================
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

# ================================
# Helper: run TFLite model
# ================================
def run_tflite(model_path, X):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    in_det  = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_scale, in_zero = in_det["quantization"]
    out_scale, out_zero = out_det["quantization"]

    scores = []

    for x in X:
        if in_det["dtype"] == np.int8:
            xq = np.round(x / in_scale + in_zero).astype(np.int8)
            xq = np.clip(xq, -128, 127)
        else:
            xq = x.astype(np.float32)

        interpreter.set_tensor(in_det["index"], xq[None, ...])
        interpreter.invoke()

        y = interpreter.get_tensor(out_det["index"])[0, 0]

        if out_det["dtype"] == np.int8:
            score = (y - out_zero) * out_scale
        else:
            score = float(y)

        scores.append(score)

    return np.array(scores)

# ================================
# Run both models
# ================================
print("Running TFLite FLOAT...")
scores_float = run_tflite("tof_model_float.tflite", X_test)

print("Running TFLite INT8...")
scores_int8 = run_tflite("tof_model_int8.tflite", X_test)

y_float = (scores_float >= THRESHOLD).astype(int)
y_int8  = (scores_int8  >= THRESHOLD).astype(int)

# ================================
# Metrics
# ================================
def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),
        "rec": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

m_f = metrics(Y_test, y_float)
m_i = metrics(Y_test, y_int8)

print("\n=== TEST SET RESULTS (TFLite) ===")
print("FLOAT:", m_f)
print("INT8 :", m_i)

# ================================
# Confusion matrices
# ================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay(
    confusion_matrix(Y_test, y_float),
    display_labels=["Invalid", "Valid"]
).plot(ax=ax[0], colorbar=False)
ax[0].set_title("TFLite FLOAT")

ConfusionMatrixDisplay(
    confusion_matrix(Y_test, y_int8),
    display_labels=["Invalid", "Valid"]
).plot(ax=ax[1], colorbar=False)
ax[1].set_title("TFLite INT8")

plt.tight_layout()
plt.show()