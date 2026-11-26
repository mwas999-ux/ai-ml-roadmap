# week7/day3/day3.py
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, f1_score
from xgboost import XGBClassifier

MON = "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
WED = "data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"


def load(path, n, seed=42):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    return df.sample(n, random_state=seed)


df = pd.concat([load(MON, 60000), load(WED, 150000)], ignore_index=True)

X = df.drop(columns=["Label"])
# XGBoost needs numbers, not text labels
le = LabelEncoder()
y = le.fit_transform(df["Label"])
print("Label codes:", dict(zip(le.classes_, range(len(le.classes_)))))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    ),
}

recalls = {}
summary = []
for name, model in models.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_s = time.time() - t0
    pred = model.predict(X_te)
    recalls[name] = recall_score(y_te, pred, average=None)
    summary.append(
        [name, round(f1_score(y_te, pred, average="macro"), 4), round(train_s, 1)]
    )

print("\n=== Overall ===")
print(pd.DataFrame(summary, columns=["model", "macro_F1", "train_seconds"]))

print("\n=== Recall per attack type ===")
print(pd.DataFrame(recalls, index=le.classes_).round(4))
