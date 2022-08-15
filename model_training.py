import tensorflow as tf
from tensorflow import keras
from img_scaling import train_ds, val_ds

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Activation('relu'))

classifier.add(Dropout(0.5))
classifier.add(Dense(1))

classifier.add(Activation('sigmoid'))

classifier.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

classifier.summary()

history = classifier.fit(
    train_ds,
    steps_per_epoch=625,
    epochs=3,
    validation_data=val_ds,
    validation_steps=5000
)

#model.summary()
