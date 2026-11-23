import numpy as np
import pandas as pd

FEATURES = [
    "Fwd Packet Length Mean",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
]

monday = pd.read_csv("data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
monday.columns = monday.columns.str.strip()
monday = monday.replace([np.inf, -np.inf], np.nan).dropna()

scan = pd.read_csv("data/portscan_clean.csv")
scan.columns = scan.columns.str.strip()

print("MONDAY (normal) median:")
print(monday[FEATURES].median())
print("\nPORTSCAN median:")
print(scan[scan["Label"] == "PortScan"][FEATURES].median())
print(
    "\nShare of Monday flows with 2 or fewer forward packets:",
    f"{(monday['Total Fwd Packets'] <= 2).mean() * 100:.1f}%",
)
