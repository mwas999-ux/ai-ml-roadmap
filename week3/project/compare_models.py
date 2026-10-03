import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- LOAD + CLEAN ----
df = sns.load_dataset("titanic")
df = df.drop(columns=["deck"])
df["age"] = df["age"].fillna(df["age"].median())
df = df.dropna(subset=["embarked"])
df["sex_encoded"] = df["sex"].map({"male": 0, "female": 1})

# ---- FEATURE ENGINEERING (from Day 3) ----
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)

# ---- ENCODING (one-hot for embarked, from Day 4) ----
embarked_dummies = pd.get_dummies(df["embarked"], prefix="embarked")
df = pd.concat([df, embarked_dummies], axis=1)

features = [
    "pclass",
    "sex_encoded",
    "age",
    "family_size",
    "is_alone",
    "fare",
    "embarked_C",
    "embarked_Q",
    "embarked_S",
]
X = df[features]
y = df["survived"]

# ---- SCALING (needed for Logistic Regression, optional for trees) ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- SPLIT for precision/recall/F1 (single split, for a confusion-matrix-style check) ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
}

print(
    f"{'Model':<22}{'CV Avg Acc':<12}{'CV Std':<10}{'Precision':<12}{'Recall':<10}{'F1':<8}"
)
print("-" * 75)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(
        f"{name:<22}{cv_scores.mean():<12.4f}{cv_scores.std():<10.4f}{precision:<12.4f}{recall:<10.4f}{f1:<8.4f}"
    )
