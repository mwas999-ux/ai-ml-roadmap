import pandas as pd

# Change this to match your exact file name
FILE = "data/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

df = pd.read_csv(FILE)

print("SHAPE (rows, columns):", df.shape)
print("MEMORY (MB):", round(df.memory_usage(deep=True).sum() / 1e6, 1))

print("\nFIRST 10 COLUMN NAMES (repr shows hidden spaces):")
print([repr(c) for c in df.columns[:10]])

print("\nLABEL COUNTS:")
print(df[" Label"].value_counts())

print("\nA FEW IMPORTANT COLUMNS:")
cols = [
    " Destination Port",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    " Label",
]
print(df[cols].head(8))
