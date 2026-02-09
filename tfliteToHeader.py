import sys
import os

def convert(tflite_path, out_h_path, var_name):
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(out_h_path, "w") as f:
        f.write("#pragma once\n\n")
        f.write(f"// Generated from {tflite_path}\n")
        f.write(f"const unsigned char {var_name}[] = {{\n")

        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11:
                f.write("\n")

        f.write("\n};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")

    print(f"Generated {out_h_path} ({len(data)} bytes)")

# ================================
# Main
# ================================

convert("tof_model_float.tflite", "model_float.h", "tof_model_float_tflite")
convert("tof_model_int8.tflite",  "model_int8.h",  "tof_model_int8_tflite")

print("Done.")
