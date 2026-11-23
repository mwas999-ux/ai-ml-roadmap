import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = [
    "Fwd Packet Length Mean",
    "Bwd Packets/s",
    "Flow IAT Max",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
]
KEEP = set(FEATURES) | {"Label"}


def load(path, n):
    df = pd.read_csv(path, usecols=lambda c: c.strip() in KEEP)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.sample(min(n, len(df)), random_state=42)


monday = load("data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv", 100_000)
wed = load("data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv", 150_000)

print("Wednesday sample labels:")
print(wed["Label"].value_counts(), "\n")

model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(monday[FEATURES])

wed["flagged"] = model.predict(wed[FEATURES]) == -1
print("Isolation Forest flag rate by label (%):")
print((wed.groupby("Label")["flagged"].mean() * 100).round(1))
