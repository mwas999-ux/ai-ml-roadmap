import numpy as np

actual_prices = np.array([100, 150, 200, 250])
model_guesses = np.array([90, 140, 220, 230])

errors = model_guesses - actual_prices
print("Errors:", errors)

squared_errors = errors**2
print("Squared errors:", squared_errors)

mse = squared_errors.mean()
print("Mean Squared Error (loss):", mse)
