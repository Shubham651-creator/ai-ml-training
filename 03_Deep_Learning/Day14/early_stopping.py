import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Dataset
# -----------------------------
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# -----------------------------
# Features & Target
# -----------------------------
X = df[["Hours_Studied"]]
y = df["Pass"]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train / Validation Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Validation Samples:", len(X_val))

# -----------------------------
# Early Stopping
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Neural Network
# -----------------------------
model = Sequential([
    Input(shape=(1,)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# Prediction
# -----------------------------
new_students = np.array([
    [2],
    [4.5],
    [6]
])

new_students_scaled = scaler.transform(
    new_students
)

predictions = model.predict(
    new_students_scaled
)

print("\nPredictions:")
for hour, pred in zip(new_students.flatten(), predictions):
    print(
        f"Hours Studied = {hour} --> Pass Probability = {pred[0]:.2f}"
    )

# -----------------------------
# Model Summary
# -----------------------------
print("\nModel Summary:")
model.summary()

# -----------------------------
# Loss Graph
# -----------------------------
plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()