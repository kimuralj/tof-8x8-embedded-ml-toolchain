import numpy as np

bg = np.load("background.npy").astype(np.float32)

assert bg.shape == (8, 8)

with open("background.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("static const float background[8][8] = {\n")

    for y in range(8):
        f.write("    { ")
        for x in range(8):
            f.write(f"{bg[y, x]:.6f}f")
            if x < 7:
                f.write(", ")
        f.write(" }")
        if y < 7:
            f.write(",")
        f.write("\n")

    f.write("};\n")

print("background.h generated!")
