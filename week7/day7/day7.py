# week7/day7/day7.py
import glob
import json
import sqlite3
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "week7/day6")
from mitre_map import enrich

DB = "data/siem_logs.db"
MODEL = "data/ids_xgb.joblib"
MIN_CONF = 0.5


class IDSDetector:
    """Loads the saved IDS and turns raw flows into collector rows."""

    def __init__(self, path=MODEL):
        d = joblib.load(path)
        self.model = d["model"]
        self.classes = d["classes"]
        self.features = d["features"]

    def score(self, raw, min_conf=MIN_CONF):
        X = raw[self.features]
        proba = self.model.predict_proba(X)
        idx = proba.argmax(axis=1)
        rows = []
        for i, (k, p) in enumerate(zip(idx, proba.max(axis=1))):
            pred = self.classes[k]
            if pred == "BENIGN" or p < min_conf:
                continue
            m = enrich(pred)
            src = raw.iloc[i]
            rows.append(
                {
                    "event_type": pred,
                    "src_ip": None,  # CICIDS has no IPs
                    "dst_port": int(src["Destination Port"]),
                    "label": src["Label"],  # true label, lab only
                    "details": json.dumps(
                        {
                            "confidence": round(float(p), 3),
                            "tactic": m["tactic"],
                            "technique": f'{m["id"]} {m["name"]}',
                            "next_step": m["next"],
                        }
                    ),
                }
            )
        return rows


# ---------- load 20,000 Wednesday flows ----------
path = glob.glob("data/MachineLearningCVE/*ednesday*")[0]
df = pd.read_csv(path, encoding="latin1")
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].str.replace(r"[^A-Za-z0-9]+", " ", regex=True).str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
flows = df.sample(20000, random_state=7).reset_index(drop=True)

det = IDSDetector()
rows = det.score(flows)
print("Flows scored: 20000   IDS alerts:", len(rows))

# ---------- look at the real table first ----------
con = sqlite3.connect(DB)
cur = con.cursor()
print("\nTable columns:")
for c in cur.execute("PRAGMA table_info(events)").fetchall():
    print("  ", c[1], c[2])
print("\nOne existing row:", cur.execute("SELECT * FROM events LIMIT 1").fetchone())
print(
    "\nBefore:",
    cur.execute("SELECT source, COUNT(*) FROM events GROUP BY source").fetchall(),
)

# ---------- write only our own source ----------
cur.execute("DELETE FROM events WHERE source = 'ids'")
now = datetime.now().isoformat(timespec="seconds")
cur.executemany(
    "INSERT INTO events (ingested_at, source, event_type, src_ip, dst_port, label, details) "
    "VALUES (?, 'ids', ?, ?, ?, ?, ?)",
    [
        (now, r["event_type"], r["src_ip"], r["dst_port"], r["label"], r["details"])
        for r in rows
    ],
)
con.commit()

print(
    "After: ",
    cur.execute("SELECT source, COUNT(*) FROM events GROUP BY source").fetchall(),
)

# ---------- read it back from the collector ----------
back = pd.read_sql_query(
    "SELECT event_type, label, details FROM events WHERE source='ids'", con
)
con.close()

total = len(back)
real = int((back["label"] != "BENIGN").sum())
print(f"\nIDS alerts: {total}   real attacks: {real}   false alarms: {total - real}")
print(f"Precision: {real / total:.3f}")

print("\n=== Alert type vs true label ===")
print(pd.crosstab(back["label"], back["event_type"]))

print("\n=== Top alarm ports ===")
ports = pd.read_sql_query(
    "SELECT dst_port, COUNT(*) AS n FROM events WHERE source='ids' "
    "GROUP BY dst_port ORDER BY n DESC LIMIT 5",
    sqlite3.connect(DB),
)
print(ports.to_string(index=False))

print("\nOne stored alert:")
print(back["details"].iloc[0])
