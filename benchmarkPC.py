import time
import numpy as np
import tensorflow as tf

# ================================
# Config
# ================================
N_RUNS = 200          # number of inferences
N_WARMUP = 20         # warm-up (to ignore)
X = np.load("X_test.npy")

X = X[:N_RUNS]

# ================================
# Benchmark function
# ================================
def benchmark(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Warm-up
    for i in range(N_WARMUP):
        x = X[i % len(X)]
        interpreter.set_tensor(
            input_details["index"],
            x[None].astype(input_details["dtype"])
        )
        interpreter.invoke()

    times = []

    for i in range(N_RUNS):
        x = X[i % len(X)]

        t0 = time.perf_counter()

        interpreter.set_tensor(
            input_details["index"],
            x[None].astype(input_details["dtype"])
        )
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times)

    return {
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "fps": 1000.0 / times.mean()
    }

# ================================
# Run benchmarks
# ================================
float_res = benchmark("tof_model_float.tflite")
int8_res  = benchmark("tof_model_int8.tflite")

print("\n=== PC INFERENCE BENCHMARK ===")
print("FLOAT32:")
for k, v in float_res.items():
    print(f"  {k}: {v:.3f}")

print("\nINT8:")
for k, v in int8_res.items():
    print(f"  {k}: {v:.3f}")

print("\nSpeedup (FLOAT / INT8):",
      float_res["mean_ms"] / int8_res["mean_ms"])