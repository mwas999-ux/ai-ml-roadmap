import numpy as np

# ---- 2D ARRAYS ----
# Think of this as a grid/spreadsheet: 2 rows (candles), 4 columns (Open, High, Low, Close)
ohlc = np.array(
    [
        [1200.5, 1203.0, 1199.0, 1201.2],  # candle 1
        [1201.2, 1204.5, 1200.8, 1203.4],  # candle 2
    ]
)
print("Full grid:\n", ohlc)

# ---- INDEXING: array[row, column] ----
# Both positions start counting from 0
print("Row 0, Col 0 (Candle 1 Open):", ohlc[0, 0])
print("Row 0, Col 3 (Candle 1 Close):", ohlc[0, 3])
print("Row 1, Col 0 (Candle 2 Open):", ohlc[1, 0])

# ---- GRABBING A WHOLE ROW OR COLUMN ----
# ':' alone means "give me everything along that axis"
print("Row 0, all columns (whole candle 1):", ohlc[0, :])
print("All rows, col 0 (every Open price):", ohlc[:, 0])

# ---- BROADCASTING: single number ----
# A single number gets stretched across every element
row = np.array([1, 2, 3, 4])
print("row + 10:", row + 10)

# ---- BROADCASTING: 1D array onto 2D array ----
# The 1D array repeats down every row, matched column-by-column
adjustment = np.array([0.1, 0.2, -0.1, 0.05])  # must match number of columns (4)
print("Adjusted OHLC:\n", ohlc + adjustment)

# ---- WHY BROADCASTING FAILS: shape mismatch ----
# ohlc has 4 columns. This array only has 3 values - can't line up evenly.
# Uncomment the two lines below to see the actual error:
# bad = np.array([1, 2, 3])
# print(ohlc + bad)
