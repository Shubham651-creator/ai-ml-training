import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    GlobalAveragePooling2D
)

# 1. Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze the base model
base_model.trainable = False

# 2. Create a new model on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    BatchNormalization(),  # Add batch normalization layer
    Dense(5, activation="softmax") # CHANGED: 5 classes for daisy, dandelion, rose, sunflower, tulip
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Load the tf_flowers dataset
dataset, info = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:]'], with_info=True, as_supervised=True)
train_dataset, val_dataset = dataset[0], dataset[1]

# 4. Define Preprocessing Function (Crucial Step)
IMG_SIZE = 128
BATCH_SIZE = 4

def preprocess_img(image, label):
    # Resize the raw image to 128x128
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normalize pixel values from [0, 255] to [-1, 1] as required by MobileNetV2
    image = image / 127.5 - 1.0
    return image, label

# 5. Apply preprocessing, shuffle, batch, and prefetch for optimization
train_batches = (
    train_dataset
    .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_batches = (
    val_dataset
    .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# 6. Train the model using the prepared batches
history = model.fit(
    train_batches,       # CHANGED: Use train_batches instead of raw train_dataset
    validation_data=val_batches, # CHANGED: Use val_batches instead of raw val_dataset
    epochs=5
)

### Output after training for 5 epochs: 
# Epoch 1/5
# I0000 00:00:1781155600.983281    6702 tf_record_dataset_op.cc:396] The default buffer size is 262144, which is overridden by the user specified `buffer_size` of 8388608
# 734/734 ━━━━━━━━━━━━━━━━━━━━ 68s 82ms/step - accuracy: 0.7306 - loss: 0.7390 - val_accuracy: 0.8774 - val_loss: 0.3551
# Epoch 2/5
# 734/734 ━━━━━━━━━━━━━━━━━━━━ 54s 73ms/step - accuracy: 0.8222 - loss: 0.5391 - val_accuracy: 0.8787 - val_loss: 0.3467
# Epoch 3/5
# 734/734 ━━━━━━━━━━━━━━━━━━━━ 51s 70ms/step - accuracy: 0.8287 - loss: 0.5038 - val_accuracy: 0.8706 - val_loss: 0.3694
# Epoch 4/5
# 734/734 ━━━━━━━━━━━━━━━━━━━━ 56s 76ms/step - accuracy: 0.8498 - loss: 0.4512 - val_accuracy: 0.8815 - val_loss: 0.3395
# Epoch 5/5
# 734/734 ━━━━━━━━━━━━━━━━━━━━ 54s 74ms/step - accuracy: 0.8515 - loss: 0.4199 - val_accuracy: 0.8774 - val_loss: 0.3317
###