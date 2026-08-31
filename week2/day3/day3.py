import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x_values = np.array([-10, -5, -1, 0, 1, 5, 10])
y_values = sigmoid(x_values)

for x, y in zip(x_values, y_values):
    print(f"x={x}: sigmoid={y:.4f}")

x_smooth = np.linspace(-10, 10, 200)
plt.plot(x_smooth, sigmoid(x_smooth))
plt.axhline(0.5, color="gray", linestyle="--")
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.savefig("week2/day3/sigmoid_shape.png")
print("Saved plot")
