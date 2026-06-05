import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# CNN expects:
# (samples, height, width, channels)

X_train = X_train.reshape(
    -1, 28, 28, 1
)

X_test = X_test.reshape(
    -1, 28, 28, 1
)

print(X_train.shape)

# CNN Model
model = Sequential([
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(28,28,1)
    ),

    MaxPooling2D(
        (2,2)
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        10,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(
    X_test,
    y_test
)

print("Test Accuracy:", test_acc)