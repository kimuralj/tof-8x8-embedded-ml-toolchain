"""
Dataset split strategy:
- Train/Validation: objects listed in TRAIN_VAL_OBJECTS, with 80/20 frame split
- Test (blind): objects listed in TEST_OBJECTS, never seen during training
"""

import json
import numpy as np
import os
import random

# =============================
# Config
# =============================
DATA_FOLDER = "trainingFiles"
MAX_DIST = 400.0
MIN_DIST = 0.0
TRAIN_VAL_SPLIT = 0.8

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Define the groups here
# -----------------------------
TRAIN_VAL_OBJECTS = [
    "cube",
    "empty",
    "orangeMug",
    "yellowCup",
    "leftHand",
    "thermalBottle",
    "kettle",
    "satoru",
    "whiteCoffeeCup",
    "closedPot"
]

TEST_OBJECTS = [
    "megaminx",
    "whiteMug",
    "rightHand"
]

# =============================
# Utils
# =============================

def clean_map(m):
    m = np.array(m, dtype=np.float32)
    m[m < MIN_DIST] = MIN_DIST
    m[m > MAX_DIST] = MAX_DIST
    return m

def load_background():
    fname = os.path.join(DATA_FOLDER, "empty", "matrix_data.json")
    with open(fname) as f:
        data = json.load(f)

    acc = None
    n = 0
    for k, mat in data.items():
        m = clean_map(mat)
        if acc is None:
            acc = m.copy()
        else:
            acc += m
        n += 1
    return acc / n

# =============================
# Load and save background
# =============================

background = load_background()
print("Background loaded.")

# save background matrix
np.save("background.npy", background)
print("Background saved as background.npy")


# =============================
# Containers
# =============================

X_train, Y_train = [], []
X_val,   Y_val   = [], []
X_test,  Y_test  = [], []

# =============================
# Walk through folders
# =============================

folders = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]

for class_name in folders:
    folder_path = os.path.join(DATA_FOLDER, class_name)
    fname = os.path.join(folder_path, "matrix_data.json")

    if not os.path.exists(fname):
        continue

    name = class_name

    # # ignore empty
    # if name == "empty":
    #     continue

    # decide which group this object belongs to
    if name in TEST_OBJECTS:
        group = "test"
    elif name in TRAIN_VAL_OBJECTS:
        group = "trainval"
    else:
        print("Skipping object not listed in any group:", name)
        continue

    # define label
    invalid_keys = ["cube", "megaminx", "satoru", "closedPot", "empty", "leftHand", "rightHand"]
    is_invalid = any(k in name for k in invalid_keys)
    label = 0 if is_invalid else 1

    print(f"Loading object: {name} | group = {group} | label = {label}")

    with open(fname) as f:
        data = json.load(f)

    keys = list(data.keys())
    random.shuffle(keys)

    n = len(keys)
    split = int(TRAIN_VAL_SPLIT * n)

    if group == "trainval":
        train_keys = keys[:split]
        val_keys   = keys[split:]
        test_keys  = []
    else:
        train_keys = []
        val_keys   = []
        test_keys  = keys

    # -----------------------------
    # Process samples
    # -----------------------------
    for k in train_keys:
        mat = data[k]
        m = clean_map(mat)

        delta = background - m
        delta[delta < 0] = 0

        sample = (delta / MAX_DIST)[..., None]

        X_train.append(sample)
        Y_train.append(label)

    for k in val_keys:
        mat = data[k]
        m = clean_map(mat)

        delta = background - m
        delta[delta < 0] = 0

        sample = (delta / MAX_DIST)[..., None]

        X_val.append(sample)
        Y_val.append(label)

    for k in test_keys:
        mat = data[k]
        m = clean_map(mat)

        delta = background - m
        delta[delta < 0] = 0

        sample = (delta / MAX_DIST)[..., None]

        X_test.append(sample)
        Y_test.append(label)

# =============================
# Convert & save
# =============================

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)

X_val = np.array(X_val, dtype=np.float32)
Y_val = np.array(Y_val, dtype=np.float32)

X_test = np.array(X_test, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.float32)

print("\nShapes:")
print("Train:", X_train.shape, Y_train.shape)
print("Val  :", X_val.shape, Y_val.shape)
print("Test :", X_test.shape, Y_test.shape)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

print("\nSaved:")
print("  X_train.npy, Y_train.npy")
print("  X_val.npy,   Y_val.npy")
print("  X_test.npy,  Y_test.npy")

print("Train label distribution:")
print("0:", np.sum(Y_train == 0))
print("1:", np.sum(Y_train == 1))
