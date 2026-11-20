import pandas as pd
from scapy.all import rdpcap, IP, TCP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------- PART A: features from raw packets (your Day 2 PCAP) ----------
rows = []
for p in rdpcap("week5/day2/sample.pcap"):
    if IP in p:
        rows.append(
            {
                "src": p[IP].src,
                "dport": p[TCP].dport if TCP in p else p.dport,
                "size": len(p),
                "is_syn": int(TCP in p and p[TCP].flags == "S"),
            }
        )
pk = pd.DataFrame(rows)

per_src = pk.groupby("src").agg(
    n_packets=("size", "count"),
    n_distinct_ports=("dport", "nunique"),
    mean_size=("size", "mean"),
    min_size=("size", "min"),
    syn_count=("is_syn", "sum"),
)
per_src["syn_ratio"] = per_src["syn_count"] / per_src["n_packets"]
print("PART A: FEATURES PER SOURCE")
print(per_src.round(2))

# ---------- PART B: which CICIDS features matter? ----------
df = pd.read_csv("data/portscan_clean.csv")
df = df[df["Flow Duration"] >= 0]  # the Day 4 fix: drop negative durations
df.to_csv("data/portscan_clean.csv", index=False)
print("\nPART B: clean rows:", df.shape)

df = df.sample(40000, random_state=42)  # sample keeps the 8GB laptop happy
X = df.drop(columns="Label")
y = (df["Label"] == "PortScan").astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

print(
    classification_report(y_te, rf.predict(X_te), target_names=["BENIGN", "PortScan"])
)

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("TOP 10 FEATURES:")
print(imp.head(10).round(4))
