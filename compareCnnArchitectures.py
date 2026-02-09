import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time
import os
import random

# ================================
# Config
# ================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================================
# Load datasets
# ================================
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

X_val   = np.load("X_val.npy")
Y_val   = np.load("Y_val.npy")

# ================================
# Model definitions
# ================================

def model_baseline():
    inputs = keras.Input(shape=(8,8,1))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)

def model_baseline_nopool():
    inputs = keras.Input(shape=(8,8,1))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)

def model_gap():
    inputs = keras.Input(shape=(8,8,1))
    x = layers.Conv2D(16, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)

def model_small():
    inputs = keras.Input(shape=(8,8,1))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)

MODELS = {
    "baseline": model_baseline,
    "baselineNoMaxPool": model_baseline_nopool,
    "gap": model_gap,
    "small": model_small,
}

# ================================
# Training & evaluation
# ================================

results = []

for name, builder in MODELS.items():
    print("\n==============================")
    print("Training:", name)

    model = builder()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    t0 = time.time()

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=32,
        verbose=0
    )

    train_time = time.time() - t0

    # Evaluate on validation set (for model selection)
    y_pred = model.predict(X_val)[:,0]
    y_bin = (y_pred > 0.5).astype(np.int32)

    f1 = f1_score(Y_val, y_bin)

    params = model.count_params()

    results.append({
        "model": name,
        "params": params,
        "train_time_s": train_time,
        "f1_val": f1,
    })

    print("Test F1:", f1)

# ================================
# Save results
# ================================

df = pd.DataFrame(results)
df.to_csv("model_comparison.csv", index=False)
print("\nSaved model_comparison.csv")
print(df)
