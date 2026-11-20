import numpy as np
import pandas as pd

FILE = "data/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
df = pd.read_csv(FILE)
print("BEFORE:", df.shape)

# 1) Strip spaces from every column name
df.columns = df.columns.str.strip()
print("\nFirst 5 columns now:", list(df.columns[:5]))

# 2) Missing values
nulls = df.isnull().sum()
print("\nNULLS (only columns that have some):")
print(nulls[nulls > 0])

# 3) Infinity values (numeric columns only)
num = df.select_dtypes(include=[np.number])
infs = np.isinf(num).sum()
print("\nINFINITIES (only columns that have some):")
print(infs[infs > 0])

# 4) Duplicate rows
print("\nDUPLICATE ROWS:", df.duplicated().sum())

# 5) Negative values (suspicious for counts and durations)
neg = (num < 0).sum()
print("\nNEGATIVE VALUES (only columns that have some):")
print(neg[neg > 0])

# 6) Clean: turn inf into NaN, then drop NaN rows and duplicates
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.drop_duplicates()

print("\nAFTER:", df.shape)
print(df["Label"].value_counts())

# 7) Save the clean version (data/ is git-ignored)
df.to_csv("data/portscan_clean.csv", index=False)
print("\nSaved data/portscan_clean.csv")
