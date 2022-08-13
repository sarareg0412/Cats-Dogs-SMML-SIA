import tensorflow as tf
from tensorflow import keras
from img_scaling import train_ds, val_ds

num_classes = 2

model = keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    #optimizer=keras.optimizers.Adamax(lr=0.001),
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=3,
)