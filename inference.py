import serial
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# =============================
# Load model and background
# =============================

model = keras.models.load_model("tof_model.h5")
background = np.load("background.npy")

print("Model and background loaded.")

# =============================
# Predict function
# =============================

def predict_matrix(matrix):
    matrix = np.clip(matrix, 0, 400)

    # -------- SUBTRACT BACKGROUND --------
    delta = background - matrix
    delta[delta < 0] = 0

    # Normalize
    delta_norm = delta / 400.0

    # Shape: (1, 8, 8, 1)
    inp = np.expand_dims(delta_norm, axis=0)
    inp = np.expand_dims(inp, axis=-1)

    valid_pred = model.predict(inp, verbose=0)

    valid_score = float(valid_pred[0][0])

    return valid_score, delta

# =============================
# Serial reader
# =============================

def read_matrix_from_serial(serial_port):
    ser = serial.Serial(serial_port, baudrate=115200, timeout=1)

    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((8, 8)), cmap='gray', vmin=0, vmax=400)

    vline = ax.axvline(3.5, color='red', linewidth=2)
    ax.set_title("Waiting for data...")

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(line)

            if line.startswith("P"):
                print("Starting to read new matrix...")

                rows = []

                while len(rows) < 8:
                    if ser.in_waiting > 0:
                        row_line = ser.readline().decode('utf-8').strip()
                        row = list(map(int, row_line.split(',')))

                        if len(row) == 8:
                            print(row)
                            rows.append(row)

                matrix = np.array(rows)
                matrix = np.clip(matrix, 0, 400)

                # Predict
                valid_score, delta = predict_matrix(matrix)

                # Update image (SHOW DELTA, NOT RAW!)
                img.set_data(delta)

                # Decision
                if valid_score > 0.7:
                    title = f"VALID object | valid={valid_score:.2f}"
                    print(title)
                else:
                    title = f"INVALID scene | valid={valid_score:.2f}"
                    print(title)

                ax.set_title(title)

                fig.canvas.draw()
                fig.canvas.flush_events()

# =============================
# Main
# =============================

def main():
    serial_port = '/dev/tty.usbmodem101'  # adjust if needed
    read_matrix_from_serial(serial_port)

if __name__ == "__main__":
    main()
