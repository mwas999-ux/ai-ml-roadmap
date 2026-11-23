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

# 1. Learn "normal" from Monday (benign only)
monday = pd.read_csv("data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
monday.columns = monday.columns.str.strip()
monday = monday.replace([np.inf, -np.inf], np.nan).dropna()
train = monday.sample(100_000, random_state=42)[FEATURES]

model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(train)
print("Trained on", len(train), "benign Monday flows")

# 2. Test on the cleaned PortScan file
scan = pd.read_csv("data/portscan_clean.csv")
scan.columns = scan.columns.str.strip()
benign_test = scan[scan["Label"] == "BENIGN"][FEATURES]
attack_test = scan[scan["Label"] == "PortScan"][FEATURES]


# predict: -1 = anomaly, 1 = normal
def flag_rate(df):
    return (model.predict(df) == -1).mean() * 100


print(f"Benign flows flagged:   {flag_rate(benign_test):.1f}%")
print(f"PortScan flows flagged: {flag_rate(attack_test):.1f}%")
