import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

df = sns.load_dataset("titanic")
df = df.drop(columns=["deck"])
df["age"] = df["age"].fillna(df["age"].median())
df = df.dropna(subset=["embarked"])
df["sex_encoded"] = df["sex"].map({"male": 0, "female": 1})

features = ["pclass", "sex_encoded", "age", "sibsp", "parch", "fare"]
X = df[features]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 score:", f1_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
