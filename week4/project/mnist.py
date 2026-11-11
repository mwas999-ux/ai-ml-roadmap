from tensorflow import keras

# Load the dataset (first run downloads it, give it a minute)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print("Train images:", X_train.shape)
print("Test images:", X_test.shape)

# Scale pixels from 0-255 down to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# The network
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),  # 28x28 grid -> 784 numbers
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),  # 10 outputs, one per digit
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=5, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)
