import importlib.util
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Load the Week 5 collector by path (avoids the day7 name clash)
spec = importlib.util.spec_from_file_location("collector_w5", "week5/day7/day7.py")
w5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w5)
LogCollector = w5.LogCollector


class AnomalyDetector:
    """SIEM Component 1: autoencoder that scores flows and reports alarms."""

    def __init__(
        self, model_path="data/anomaly_ae.keras", meta_path="data/anomaly_meta.joblib"
    ):
        self.model = keras.models.load_model(model_path)
        meta = joblib.load(meta_path)
        self.scaler = meta["scaler"]
        self.threshold = meta["threshold"]
        self.features = meta["features"]

    def score(self, df):
        x = np.log1p(df[self.features].clip(lower=0).values)
        x = self.scaler.transform(x)
        return np.mean((x - self.model.predict(x, verbose=0)) ** 2, axis=1)

    def alarms_to_rows(self, df):
        """Return collector rows for every flow above the threshold."""
        df = df.copy()
        df["error"] = self.score(df)
        hits = df[df["error"] > self.threshold]
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for _, r in hits.iterrows():
            details = {
                "recon_error": round(float(r["error"]), 4),
                "threshold": round(self.threshold, 4),
                "model": "autoencoder_v1",
                "narrow_detector": "slow-connection attacks",
            }
            for f in self.features:
                details[f] = round(float(r[f]), 2)
            rows.append(
                (
                    now,
                    "anomaly",
                    "flow_anomaly",
                    None,
                    int(r["Destination Port"]),
                    r["Label"],
                    json.dumps(details),
                )
            )
        return rows, len(df)


if __name__ == "__main__":
    detector = AnomalyDetector()
    cols = set(detector.features) | {"Destination Port", "Label"}

    df = pd.read_csv(
        "data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
        usecols=lambda c: c.strip() in cols,
    )
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    batch = df.sample(20_000, random_state=7)  # new sample, different from Day 4-6

    rows, scored = detector.alarms_to_rows(batch)

    c = LogCollector()
    c.con.execute("DELETE FROM events WHERE source='anomaly'")  # only our own rows
    c.con.commit()
    c._store(rows)

    print(f"Scored {scored} flows, wrote {len(rows)} anomaly events")

    print("\nEVENTS PER SOURCE (whole collector):")
    for row in c.query(
        "SELECT source, event_type, COUNT(*) FROM events GROUP BY source, event_type"
    ):
        print(row)

    print("\nANOMALY ALARMS BY TRUE LABEL (lab check only):")
    for row in c.query(
        "SELECT label, COUNT(*) FROM events WHERE source='anomaly' "
        "GROUP BY label ORDER BY 2 DESC"
    ):
        print(row)

    print("\nTOP DESTINATION PORTS IN ALARMS:")
    for row in c.query(
        "SELECT dst_port, COUNT(*) FROM events WHERE source='anomaly' "
        "GROUP BY dst_port ORDER BY 2 DESC LIMIT 5"
    ):
        print(row)

    print("\nONE ANOMALY EVENT:")
    print(c.query("SELECT * FROM events WHERE source='anomaly' LIMIT 1")[0])
