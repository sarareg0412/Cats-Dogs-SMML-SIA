from tensorflow import keras
from img_scaling import train_dataset, val_dataset

model = keras.models.Sequential()

initializers = {

}

model.add(
    keras.layers.Conv2D(
        24, 5, input_shape=(256, 256, 3),
        activation='relu',
    )
)
model.add(keras.layers.MaxPooling2D(2))
model.add(
    keras.layers.Conv2D(
        48, 5, activation='relu',
    )
)
model.add(keras.layers.MaxPooling2D(2))
model.add(
    keras.layers.Conv2D(
        96, 5, activation='relu',
    )
)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.9))

model.add(keras.layers.Dense(
    2, activation='softmax',)
)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adamax(lr=0.001),
              metrics=['acc'])

history = model.fit(
    train_dataset,
    validation_data = val_dataset,
    workers=10,
    epochs=20,
)