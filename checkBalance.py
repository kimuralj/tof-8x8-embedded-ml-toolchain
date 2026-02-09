import numpy as np

def report(name, y):
    total = len(y)
    valid = np.sum(y == 1)
    invalid = np.sum(y == 0)
    print(f"{name}:")
    print(f"  total   = {total}")
    print(f"  valid   = {valid} ({100*valid/total:.1f}%)")
    print(f"  invalid = {invalid} ({100*invalid/total:.1f}%)")
    print()

Y_train = np.load("Y_train.npy")
Y_val   = np.load("Y_val.npy")
Y_test  = np.load("Y_test.npy")

report("TRAIN", Y_train)
report("VAL",   Y_val)
report("TEST",  Y_test)