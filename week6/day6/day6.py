import numpy as np
import pandas as pd
import joblib
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
model.fit(x_train, x_train, epochs=40, batch_size=256, verbose=0)


def errors(x):
    return np.mean((x - model.predict(x, verbose=0)) ** 2, axis=1)


val_err = errors(x_val)
wed["error"] = errors(x_wed)
wed["is_attack"] = wed["Label"] != "BENIGN"

benign = wed[~wed["is_attack"]]
attacks = wed[wed["is_attack"]].sample(int(len(benign) * 0.0101), random_state=42)
realistic = pd.concat([benign, attacks])

for pct in [99, 99.5, 99.9]:
    t = np.percentile(val_err, pct)
    fl = realistic["error"] > t
    tp = (fl & realistic["is_attack"]).sum()
    fp = (fl & ~realistic["is_attack"]).sum()
    per_10k = (tp + fp) / len(realistic) * 10_000
    print(f"--- {pct}th percentile, threshold {t:.4f} ---")
    print(
        f"alarms per 10,000 flows: {per_10k:.0f}   real={tp}  false={fp}  precision={tp/max(tp+fp,1):.2f}"
    )
    rec = (wed.assign(f=wed["error"] > t).groupby("Label")["f"].mean() * 100).round(1)
    print("recall by label (%):", rec.to_dict(), "\n")

# Save the pieces Day 7 needs (data/ is gitignored)
CHOSEN = 99.5
threshold = float(np.percentile(val_err, CHOSEN))
model.save("data/anomaly_ae.keras")
joblib.dump(
    {"scaler": scaler, "threshold": threshold, "features": FEATURES},
    "data/anomaly_meta.joblib",
)
print(f"Saved model + scaler + threshold ({CHOSEN}th pct = {threshold:.4f})")
