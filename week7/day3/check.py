# week7/day3/check.py
import pandas as pd

df = pd.read_csv(
    "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv", nrows=50000
)
df.columns = df.columns.str.strip()
cols = ["Flow Bytes/s", "Flow Packets/s"]

print("1) after read_csv:  ", df[cols].dtypes.to_dict())

df = df.replace([float("inf"), float("-inf")], pd.NA)
print("2) after replace:   ", df[cols].dtypes.to_dict())

df = df.dropna()
print("3) after dropna:    ", df[cols].dtypes.to_dict())
