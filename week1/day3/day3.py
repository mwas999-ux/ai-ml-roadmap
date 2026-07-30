import pandas as pd
import numpy as np

# ---- SERIES: a single labeled column of data ----
prices = pd.Series([1200.5, 1201.2, 1199.8, 1203.4], name="close_price")
print("Series:\n", prices)
print("\n")

# ---- DATAFRAME: a full labeled table (like ohlc, but with column names) ----
data = {
    "open": [1200.5, 1201.2, 1203.4],
    "high": [1203.0, 1204.5, 1206.0],
    "low": [1199.0, 1200.8, 1202.0],
    "close": [1201.2, 1203.4, 1205.5],
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)
print("\n")

# ---- BASIC INSPECTION ----
print("Shape:", df.shape)  # (rows, columns)
print("Columns:", df.columns.tolist())
print("\nFirst 2 rows:\n", df.head(2))
print("\nInfo:")
df.info()
print("\nSummary stats:\n", df.describe())
