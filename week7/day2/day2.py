# week7/day2/day2.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MON = "data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
WED = "data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"


def load(path, n, seed=42):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    df = df.drop_duplicates()
    return df.sample(n, random_state=seed)


mon = load(MON, 60000)
wed = load(WED, 150000)
df = pd.concat([mon, wed], ignore_index=True)

print("Label counts after loading:")
print(df["Label"].value_counts())

X = df.drop(columns=["Label"])
y = df["Label"]

# stratify keeps the rare classes present in both train and test
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

print("\n=== Random Forest, per attack type (same-day split) ===")
print(classification_report(y_te, rf.predict(X_te), zero_division=0))

imp = pd.Series(rf.feature_importances_, index=X.columns)
print("Top 5 features:")
print(imp.sort_values(ascending=False).head(5))
