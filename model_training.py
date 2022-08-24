import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow import keras
from preprocessing import train_dataset, val_dataset, BATCH_SIZE, IMGS_PATH, SIZE, STEPS_PER_EPOCH

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_testing import  plot_scores

import pandas as pd
import numpy as np

MAX_BATCHES = 25000/BATCH_SIZE

def get_model(i: int):
    if i == 1:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dropout(0.5))
        model.add(Dense(1))

        model.add(Activation('sigmoid'))

        adam_optimizer = tf.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=adam_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # model.summary()
        return model

# history = get_model(1).fit(
#     train_ds,
#     epochs=3,
#     validation_data=val_ds,
#     callbacks=[mcp_save, reduce_lr_loss, early_stopping],
# )

N_OF_FOLDS = 5
N_OF_EPOCHS = 3

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')


def get_model_name(i):
    return 'model_'+str(i)+'.hdf5'

def k_fold_cross_validation(model_index):

    HISTORIES = []

    #print(f"Number of samples: {len(X)} \nNumber of photos: {len(train_dataset)}")

    model = get_model(model_index)
    X_train, y_train, X_test, y_test = [], [], [], []
    n_fold = 1
    k_fold = KFold(n_splits=N_OF_FOLDS, shuffle= True)

    #Kfold training loop
    for train, test in k_fold.split(train_dataset):
        print('------------------------------------------------------------------------')
        print(f'Training fold {n_fold} ...')

        #Create callback to save model for current fold
        mcp_save = ModelCheckpoint(get_model_name(model_index),
                                   save_best_only=True,
                                   monitor='val_loss',
                                   mode='min',
                                   verbose=1)

        # Training images and training labels
        i = 0
        for image_batch in train_dataset:
            if i > MAX_BATCHES:
                break
            else:
                i +=1
                if train_dataset.batch_index in train:
                    X_train.append(image_batch[0])
                    y_train.append(image_batch[1])
                else:
                    X_test.append(image_batch[0])
                    y_test.append(image_batch[1])


        history = model.fit(
            X_train,                                  # Training data
            y_train,                                  # Training labels
            validation_data=(X_test, y_test),                      # Validation set
            epochs=N_OF_EPOCHS,
            callbacks=[mcp_save, reduce_lr_loss, early_stopping],
            steps_per_epoch=STEPS_PER_EPOCH,
            verbose=1
        )

        HISTORIES.append(history)
        n_fold += 1

    return HISTORIES


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
plot_scores(k_fold_cross_validation(1), 1)

