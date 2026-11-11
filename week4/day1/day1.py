import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# ---- SAME CLEANED TITANIC DATA AS BEFORE ----
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

# Neural networks need scaled data to train well
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- BUILD THE NEURAL NETWORK ----
model = keras.Sequential(
    [
        keras.layers.Dense(
            8, activation="relu", input_shape=(6,)
        ),  # Layer 1: 8 neurons
        keras.layers.Dense(4, activation="relu"),  # Layer 2: 4 neurons
        keras.layers.Dense(1, activation="sigmoid"),  # Output: 1 neuron, probability
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=30,
    validation_data=(X_test_scaled, y_test),
    verbose=1,
)
