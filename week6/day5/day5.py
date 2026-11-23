import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

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


def prep(df, scaler=None):
    x = np.log1p(df[FEATURES].clip(lower=0).values)
    if scaler is None:
        scaler = StandardScaler().fit(x)
    return scaler.transform(x), scaler


monday = load("data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv", 100_000)
wed = load("data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv", 150_000)
x_mon, scaler = prep(monday)
x_wed, _ = prep(wed, scaler)
x_train, x_val = x_mon[:80_000], x_mon[80_000:]

model = keras.Sequential(
    [
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(12, activation="relu"),
        keras.layers.Dense(3, activation="relu"),
        keras.layers.Dense(12, activation="relu"),
        keras.layers.Dense(6),
    ]
)
model.compile(optimizer="adam", loss="mse")
model.fit(x_train, x_train, epochs=20, batch_size=256, verbose=0)


def errors(x):
    return np.mean((x - model.predict(x, verbose=0)) ** 2, axis=1)


val_err = errors(x_val)
wed["error"] = errors(x_wed)
wed["is_attack"] = wed["Label"] != "BENIGN"


def score(df, threshold):
    flagged = df["error"] > threshold
    tp = (flagged & df["is_attack"]).sum()
    fp = (flagged & ~df["is_attack"]).sum()
    fn = (~flagged & df["is_attack"]).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return tp, fp, precision, recall


# Realistic world: all benign + attacks at about 1% of the total
benign = wed[~wed["is_attack"]]
attacks = wed[wed["is_attack"]].sample(int(len(benign) * 0.0101), random_state=42)
realistic = pd.concat([benign, attacks])

print(f"Sample world: {len(wed)} flows, {wed['is_attack'].mean()*100:.0f}% attacks")
print(
    f"Realistic world: {len(realistic)} flows, {realistic['is_attack'].mean()*100:.1f}% attacks\n"
)

for pct in [99, 95, 90]:
    t = np.percentile(val_err, pct)
    print(f"--- Threshold = {pct}th percentile of normal error ({t:.4f}) ---")
    for name, df in [("Sample   ", wed), ("Realistic", realistic)]:
        tp, fp, p, r = score(df, t)
        print(
            f"{name}: alarms={tp+fp:6d}  real={tp:6d}  false={fp:6d}  precision={p:.2f}  recall={r:.2f}"
        )
    print()
