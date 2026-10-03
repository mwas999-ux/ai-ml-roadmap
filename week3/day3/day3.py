import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = sns.load_dataset("titanic")
df = df.drop(columns=["deck"])
df["age"] = df["age"].fillna(df["age"].median())
df = df.dropna(subset=["embarked"])
df["sex_encoded"] = df["sex"].map({"male": 0, "female": 1})

features = ["pclass", "sex_encoded", "age", "sibsp", "parch", "fare"]
X = df[features]
y = df["survived"]

model = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(model, X, y, cv=5)
print("5 individual scores:", scores)
print("Average accuracy:", scores.mean())
print("Standard deviation:", scores.std())
# New feature: total family size aboard
df["family_size"] = df["sibsp"] + df["parch"] + 1  # +1 for the person themselves

# New feature: were they alone?
df["is_alone"] = (df["family_size"] == 1).astype(int)

print(df[["sibsp", "parch", "family_size", "is_alone"]].head(10))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Compare: original features vs original + new family features
original_features = ["pclass", "sex_encoded", "age", "sibsp", "parch", "fare"]
new_features = ["pclass", "sex_encoded", "age", "family_size", "is_alone", "fare"]

model = RandomForestClassifier(n_estimators=100, random_state=42)

original_scores = cross_val_score(model, df[original_features], df["survived"], cv=5)
new_scores = cross_val_score(model, df[new_features], df["survived"], cv=5)

print("Original features avg:", original_scores.mean())
print("New engineered features avg:", new_scores.mean())
