import numpy as np

rng = np.random.default_rng(42)
x = rng.uniform(0, 10, 200)
normal = np.column_stack([x, 2 * x + rng.normal(0, 0.3, 200)])
odd = np.array([[3, 10], [5, 4], [8, 2]])


def rebuild(p):
    z = p[:, 0]  # encoder: keep one number
    return np.column_stack([z, 2 * z])  # decoder: rebuild both


def error(p):
    return ((p - rebuild(p)) ** 2).mean(axis=1)


print(
    f"Normal points: mean error {error(normal).mean():.3f}, max {error(normal).max():.3f}"
)
for p, e in zip(odd, error(odd)):
    print(f"Odd point {p}: rebuilt as {rebuild(p[None, :])[0]}, error {e:.2f}")
