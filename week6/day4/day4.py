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
    x = np.log1p(df[FEATURES].clip(lower=0).values)  # squash the long tail
    if scaler is None:
        scaler = StandardScaler().fit(x)  # learn from Monday only
    return scaler.transform(x), scaler


monday = load("data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv", 100_000)
wed = load("data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv", 150_000)

x_mon, scaler = prep(monday)
x_wed, _ = prep(wed, scaler)

# 80k to learn, 20k held back to set the threshold
x_train, x_val = x_mon[:80_000], x_mon[80_000:]

model = keras.Sequential(
    [
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(12, activation="relu"),
        keras.layers.Dense(3, activation="relu"),  # the squeeze
        keras.layers.Dense(12, activation="relu"),
        keras.layers.Dense(6),  # rebuild
    ]
)
model.compile(optimizer="adam", loss="mse")
model.fit(
    x_train,
    x_train,
    epochs=20,
    batch_size=256,
    validation_data=(x_val, x_val),
    verbose=2,
)


def errors(x):
    return np.mean((x - model.predict(x, verbose=0)) ** 2, axis=1)


val_err = errors(x_val)
threshold = np.percentile(val_err, 99)  # top 1% of normal = alarm
print(f"\nThreshold (99th percentile of normal error): {threshold:.4f}")
print(f"Median normal error: {np.median(val_err):.4f}")

wed["error"] = errors(x_wed)
wed["flagged"] = wed["error"] > threshold
print("\nAutoencoder flag rate by label (%):")
print((wed.groupby("Label")["flagged"].mean() * 100).round(1))
print("\nMedian error by label:")
print(wed.groupby("Label")["error"].median().round(4))
