# week7/day1/day1.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FILE = "data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"
KEEP = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Fwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "Label"]

df = pd.read_csv(FILE, usecols=lambda c: c.strip() in KEEP)
df.columns = df.columns.str.strip()
df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
df = df.sample(60000, random_state=42)

# 1) Traditional IDS: one hand-written rule
# "Lots of forward packets and nothing coming back = attack"
rule_flag = (df["Total Fwd Packets"] > 100) & (df["Total Backward Packets"] == 0)
df["rule_says_attack"] = rule_flag
df["is_attack"] = df["Label"] != "BENIGN"

print("=== TRADITIONAL RULE IDS ===")
print(pd.crosstab(df["Label"], df["rule_says_attack"]))

# 2) ML IDS: Random Forest learns from labels
X = df[KEEP[:-1]]
y = df["is_attack"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
print("\n=== ML IDS (Random Forest, same day) ===")
print(classification_report(y_te, rf.predict(X_te), target_names=["BENIGN", "ATTACK"]))