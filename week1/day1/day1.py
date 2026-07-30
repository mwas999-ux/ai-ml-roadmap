import numpy as np

prices_np = np.array([100, 102, 98, 105])
print(prices_np > 100)

prices = np.array([1200.5, 1201.2, 1199.8, 1203.4, 1205.0])

print(prices)
print("Shapes:", prices.shape)
print("Data type:", prices.dtype)

ohlc = np.array(
    [
        [1200.5, 1203.0, 1199.0, 1201.2],
        [1201.2, 1204.5, 1200.8, 1203.4],
    ]
)

print("Shape:", ohlc.shape)

returns = np.diff(prices) / prices[:-1] * 100
print("% returns:", returns)


print("Mean:", prices.mean())
print("Std dev:", prices.std())
