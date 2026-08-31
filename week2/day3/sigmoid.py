import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x_values = np.array([-10, -5, -1, 0, 1, 5, 10])
y_values = sigmoid(x_values)

for x, y in zip(x_values, y_values):
    print(f"x={x}: sigmoid={y:.4f}")
