import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import tensorflow as tf

# ================================
# Reproducibility
# ================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================================
# Load data
# ================================
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_val   = np.load("X_val.npy")
Y_val   = np.load("Y_val.npy")

# ================================
# Model
# ================================
def model_final():
    inputs = keras.Input(shape=(8,8,1))
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out)

model = model_final()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================
# Callbacks
# ================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True
)

# ================================
# Train
# ================================
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=60,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ================================
# Save model
# ================================
model.save("tof_model.h5")
print("Saved tof_model.h5")

# ================================
# Plot training curves
# ================================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()