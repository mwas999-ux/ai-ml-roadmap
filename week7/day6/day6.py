# week7/day6/day6.py
import glob
import numpy as np
import pandas as pd
import joblib
from mitre_map import enrich

d = joblib.load("data/ids_xgb.joblib")
model, classes, features = d["model"], d["classes"], d["features"]

PATTERNS = [
    "*Monday*",
    "*Tuesday*",
    "*WebAttacks*",
    "*Friday-WorkingHours-Morning*",
    "*PortScan*",
]
frames = []
for p in PATTERNS:
    path = glob.glob("data/MachineLearningCVE/" + p)[0]
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    frames.append(df.sample(400, random_state=1))
data = pd.concat(frames, ignore_index=True)

X = data[features]
proba = model.predict_proba(X)
idx = proba.argmax(axis=1)
data["predicted"] = [classes[i] for i in idx]
data["confidence"] = proba.max(axis=1).round(3)

alerts = data[(data["predicted"] != "BENIGN") & (data["confidence"] >= 0.5)].copy()
print("Flows scored:", len(data), " Alerts raised:", len(alerts))

# one example alert per predicted type
print("\n=== One enriched alert per type ===")
for lab, g in alerts.groupby("predicted"):
    r = g.iloc[0]
    m = enrich(lab)
    print(f"[{m['tactic']}] {lab} -> {m['id']} {m['name']}  (conf {r['confidence']})")
    print(f"    next step: {m['next']}")

# how many alerts per tactic
alerts["tactic"] = alerts["predicted"].map(lambda l: enrich(l)["tactic"])
print("\n=== Alerts per tactic ===")
print(alerts["tactic"].value_counts())
