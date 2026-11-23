import numpy as np
import pandas as pd

FEATURE = "Fwd Packet Length Mean"

# 1. Baseline: Monday = benign only
monday = pd.read_csv("data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
monday.columns = monday.columns.str.strip()
monday = monday.replace([np.inf, -np.inf], np.nan).dropna()
monday = monday.sample(100_000, random_state=42)  # 8GB laptop

mean = monday[FEATURE].mean()
std = monday[FEATURE].std()
print(f"Baseline ({FEATURE}): mean={mean:.2f}, std={std:.2f}")

# 2. Test data: your cleaned PortScan file
scan = pd.read_csv("data/portscan_clean.csv")
scan.columns = scan.columns.str.strip()
benign_test = scan[scan["Label"] == "BENIGN"]
attack_test = scan[scan["Label"] == "PortScan"]


# 3. z-score and flag
def flag_rate(df):
    z = (df[FEATURE] - mean) / std
    return (z.abs() > 3).mean() * 100


print(f"Benign flows flagged as anomalies:   {flag_rate(benign_test):.1f}%")
print(f"PortScan flows flagged as anomalies: {flag_rate(attack_test):.1f}%")
