import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# -----------------------------
# Load Dataset
# -----------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()    

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("\nImage Shape:", X_train.shape)
print("\nImage [0]Shape:", X_train[0].shape)
print("First Image:", X_train[0])
# -----------------------------
# Display first image
# -----------------------------
plt.imshow(X_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
 
plt.show()