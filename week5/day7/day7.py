import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP

DB = "data/siem_logs.db"


def clean(df):
    """Day 4 cleaning, wrapped in one function so every source gets washed the same way."""
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    if "Flow Duration" in df.columns:
        df = df[df["Flow Duration"] >= 0]
    return df


class LogCollector:
    def __init__(self, db_path=DB):
        self.con = sqlite3.connect(db_path)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ingested_at TEXT, source TEXT, event_type TEXT,
                src_ip TEXT, dst_port INTEGER, label TEXT, details TEXT)
        """)
        self.con.commit()

    def reset(self):
        self.con.execute("DELETE FROM events")
        self.con.commit()

    def _store(self, rows):
        self.con.executemany(
            "INSERT INTO events (ingested_at, source, event_type, src_ip, dst_port, label, details) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.con.commit()

    def ingest_pcap(self, path):
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for p in rdpcap(path):
            if IP not in p:
                continue
            layer = p[TCP] if TCP in p else p[UDP]
            details = {
                "dst_ip": p[IP].dst,
                "proto": "TCP" if TCP in p else "UDP",
                "size": len(p),
            }
            rows.append(
                (
                    now,
                    "pcap",
                    "packet",
                    p[IP].src,
                    int(layer.dport),
                    None,
                    json.dumps(details),
                )
            )
        self._store(rows)
        return len(rows)

    def ingest_cicids(self, path, n=1000):
        now = datetime.now().isoformat(timespec="seconds")
        df = clean(pd.read_csv(path)).sample(n, random_state=42)
        rows = []
        for _, r in df.iterrows():
            details = {
                "duration_us": float(r["Flow Duration"]),
                "fwd_packets": int(r["Total Fwd Packets"]),
                "avg_packet_size": round(float(r["Average Packet Size"]), 2),
            }
            rows.append(
                (
                    now,
                    "cicids",
                    "flow",
                    None,
                    int(r["Destination Port"]),
                    r["Label"],
                    json.dumps(details),
                )
            )
        self._store(rows)
        return len(rows)

    def query(self, sql):
        return self.con.execute(sql).fetchall()


if __name__ == "__main__":
    c = LogCollector()
    c.reset()  # start fresh each run so reruns don't pile up duplicates

    print("PCAP events added:  ", c.ingest_pcap("week5/day2/sample.pcap"))
    print("CICIDS events added:", c.ingest_cicids("data/portscan_clean.csv", n=1000))

    print("\nEVENTS PER SOURCE:")
    for row in c.query(
        "SELECT source, event_type, COUNT(*) FROM events GROUP BY source, event_type"
    ):
        print(row)

    print("\nCICIDS LABELS IN THE COLLECTOR:")
    for row in c.query(
        "SELECT label, COUNT(*) FROM events WHERE source='cicids' GROUP BY label"
    ):
        print(row)

    print("\nWHO KNOCKED ON THE MOST DIFFERENT DOORS (pcap):")
    for row in c.query(
        "SELECT src_ip, COUNT(DISTINCT dst_port) FROM events "
        "WHERE source='pcap' GROUP BY src_ip ORDER BY 2 DESC"
    ):
        print(row)

    print("\nONE STORED EVENT:")
    print(c.query("SELECT * FROM events WHERE source='cicids' LIMIT 1")[0])
