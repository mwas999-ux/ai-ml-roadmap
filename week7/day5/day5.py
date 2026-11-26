# week7/day5/day5.py
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

CAP = 8000
TRAIN_FRAC = 0.7
MODEL_PATH = "data/ids_xgb.joblib"
META = ["Label", "order", "is_friday"]


class IDSPipeline:
    """Raw flows in -> alert rows out. Can be saved and loaded."""

    def fit(self, X, labels):
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.features = list(X.columns)
        self.classes = list(le.classes_)
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X, y)

    def save(self, path):
        joblib.dump(
            {"model": self.model, "classes": self.classes, "features": self.features},
            path,
        )

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        p = cls()
        p.model, p.classes, p.features = d["model"], d["classes"], d["features"]
        return p

    def predict(self, raw, min_conf=0.5):
        df = raw.copy()
        df.columns = df.columns.str.strip()
        X = df[self.features].replace([np.inf, -np.inf], np.nan).dropna()
        proba = self.model.predict_proba(X)
        idx = proba.argmax(axis=1)
        out = pd.DataFrame(
            {
                "row": X.index,
                "predicted": [self.classes[i] for i in idx],
                "confidence": proba.max(axis=1).round(3),
            }
        )
        out["is_alert"] = (out["predicted"] != "BENIGN") & (
            out["confidence"] >= min_conf
        )
        return out


def load_one(path):
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].str.replace(r"[^A-Za-z0-9]+", " ", regex=True).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    df = df.reset_index(drop=True)
    df["order"] = df.index  # file order is roughly time order
    df["is_friday"] = "Friday" in path
    parts = [
        g.sample(min(len(g), CAP), random_state=42) for _, g in df.groupby("Label")
    ]
    return pd.concat(parts).sort_values("order")


# ---------- build train and test ----------
train_parts, test_parts = [], []
for f in sorted(glob.glob("data/MachineLearningCVE/*.csv")):
    d = load_one(f)
    print("loaded", f.split("\\")[-1].split("/")[-1], len(d))
    for lab, g in d.groupby("Label"):
        if lab == "BENIGN":
            # benign: Friday is the unseen day, all other days train
            (test_parts if g["is_friday"].iloc[0] else train_parts).append(g)
        else:
            cut = int(len(g) * TRAIN_FRAC)
            train_parts.append(g.iloc[:cut])
            test_parts.append(g.iloc[cut:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)
print("\ntrain:", len(train), " test:", len(test))
print("\nTest label counts:")
print(test["Label"].value_counts())

# ---------- train and save ----------
pipe = IDSPipeline()
pipe.fit(train.drop(columns=META), train["Label"])
pipe.save(MODEL_PATH)
print("\nSaved model to", MODEL_PATH)

# ---------- evaluate ----------
out = pipe.predict(test)
assert len(out) == len(test), "some test rows were dropped"
print("\n=== Time-ordered attacks + unseen-day benign ===")
print(classification_report(test["Label"], out["predicted"], zero_division=0))

# ---------- alert budget on unseen-day benign ----------
fri_benign = test[test["Label"] == "BENIGN"]
for mc in (0.5, 0.9):
    o = pipe.predict(fri_benign, min_conf=mc)
    print(
        f"Friday benign, min_conf {mc}: {o['is_alert'].sum()} false alarms "
        f"in {len(o)} flows = {o['is_alert'].mean() * 10000:.1f} per 10,000"
    )

# ---------- demo: one raw flow per class, then reload from disk ----------
demo = test.groupby("Label").head(1)
alerts = pipe.predict(demo)
alerts.insert(0, "true_label", demo["Label"].values)
print("\n=== Demo: one flow per class ===")
print(
    alerts[["true_label", "predicted", "confidence", "is_alert"]].to_string(index=False)
)

reloaded = IDSPipeline.load(MODEL_PATH)
same = (reloaded.predict(demo)["predicted"].values == alerts["predicted"].values).all()
print("\nReloaded model gives the same answers:", same)
