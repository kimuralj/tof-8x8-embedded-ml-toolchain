import tensorflow as tf
import numpy as np

# ================================
# Load model
# ================================
model = tf.keras.models.load_model("tof_model.h5")

# ================================
# Load representative data
# ================================
# This must be saved during training:
# np.save("X_train.npy", X)
X = np.load("X_train.npy").astype(np.float32)

print("Loaded X_train:", X.shape, X.dtype, "range:", X.min(), X.max())

# ================================
# --------- EXPORT FLOAT --------
# ================================
print("\nExporting FLOAT model...")

converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
converter_float.optimizations = []   # no quantization

tflite_float = converter_float.convert()

with open("tof_model_float.tflite", "wb") as f:
    f.write(tflite_float)

print("Saved tof_model_float.tflite | size:", len(tflite_float)/1024, "KB")

# Inspect
interpreter = tf.lite.Interpreter(model_path="tof_model_float.tflite")
interpreter.allocate_tensors()
print("FLOAT input :", interpreter.get_input_details())
print("FLOAT output:", interpreter.get_output_details())

# ================================
# --------- EXPORT INT8 ---------
# ================================
print("\nExporting INT8 model...")

def representative_dataset():
    # Use real samples
    for i in range(min(500, len(X))):
        yield [X[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8 = converter_int8.convert()

with open("tof_model_int8.tflite", "wb") as f:
    f.write(tflite_int8)

print("Saved tof_model_int8.tflite | size:", len(tflite_int8)/1024, "KB")

# Inspect
interpreter = tf.lite.Interpreter(model_path="tof_model_int8.tflite")
interpreter.allocate_tensors()
print("INT8 input :", interpreter.get_input_details())
print("INT8 output:", interpreter.get_output_details())

print("\nDone.")
