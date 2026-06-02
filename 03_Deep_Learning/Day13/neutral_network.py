import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

# Load the data
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8],
    "Pass": [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

print(df)

# Features
X = df[["Hours_Studied"]]
y = df["Pass"]

# Create netural network model
model = Sequential([
    Input(shape=(1,)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

history = model.fit(
    X_scaled,
    y,
    epochs=100,
    verbose=1
)

# predictions = model.predict(X_scaled)
# for hour, pred in zip(df["Hours_Studied"], predictions):
#     print(hour, pred[0])

predictions = model.predict(
    scaler.transform(
        np.array([[2],[6]])
    )
)

print(predictions)

print(model.summary())