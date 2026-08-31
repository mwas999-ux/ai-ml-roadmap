import pandas as pd
import seaborn as sns

# Titanic dataset is built into seaborn - no download needed
df = sns.load_dataset("titanic")

print("Shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())
# Drop 'deck' - too sparse to be useful (77% missing)
df = df.drop(columns=["deck"])

# Fill 'age' with the median (more robust to outliers than mean)
df["age"] = df["age"].fillna(df["age"].median())

# Drop the 2 rows missing 'embarked' - small number, safe to drop
df = df.dropna(subset=["embarked"])

print("\nShape after cleaning:", df.shape)
print("Missing values after cleaning:")
print(df.isnull().sum())
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Select numeric-friendly features (sex needs converting - it's text)
df["sex_encoded"] = df["sex"].map({"male": 0, "female": 1})

features = ["pclass", "sex_encoded", "age", "sibsp", "parch", "fare"]
X = df[features]
y = df["survived"]

# Split into training data and testing data (never seen by the model during training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy on test data:", accuracy)
print("Feature importance:")
for feat, imp in zip(features, model.feature_importances_):
    print(f"  {feat}: {imp:.3f}")
    import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.barplot(x=features, y=model.feature_importances_)
plt.title("Titanic Survival - Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("week2/project/feature_importance.png")
print("\nPlot saved")
