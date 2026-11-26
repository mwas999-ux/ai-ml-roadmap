# week7/day4/day4.py
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

CAP = 8000  # max flows kept per label, per file


def load_one(path):
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.replace(r"[^A-Za-z0-9]+", " ", regex=True).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    parts = []
    for lab, g in df.groupby("Label"):
        parts.append(g.sample(min(len(g), CAP), random_state=42))
    return pd.concat(parts)


files = sorted(glob.glob("data/MachineLearningCVE/*.csv"))
print("Files found:", len(files))

frames = []
for f in files:
    part = load_one(f)
    print("loaded", f.split("/")[-1], len(part))
    frames.append(part)

df = pd.concat(frames, ignore_index=True)
# the same label can appear in more than one file (BENIGN does), so cap again
df = pd.concat(
    [g.sample(min(len(g), CAP * 2), random_state=42) for _, g in df.groupby("Label")],
    ignore_index=True,
)

print("\n=== Label counts ===")
print(df["Label"].value_counts())

X = df.drop(columns=["Label"])
le = LabelEncoder()
y = le.fit_transform(df["Label"])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
print("\nTraining on", len(X_tr), "flows...")
model.fit(X_tr, y_tr)

print("\n=== XGBoost, every attack type (shuffled split) ===")
print(
    classification_report(
        y_te, model.predict(X_te), target_names=le.classes_, zero_division=0
    )
)
