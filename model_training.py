import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow import keras
from preprocessing import train_dataset, val_dataset, BATCH_SIZE, IMGS_PATH, SIZE

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_testing import  plot_scores

import pandas as pd
import numpy as np

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
N_OF_EPOCHS = 5
SHUFFLE_BUFFER_SIZE = 100


save_dir = '/saved_models/'
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=15, verbose=0, mode='min')


def get_model_name(i,k):
    return 'model_'+str(i)+"_"+str(k)+'.h5'

def k_fold_cross_validation(i):
    #Training images and training labels
    X, y = train_dataset.next()

    # For augmentation
    traing_idg = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.3,
                             fill_mode='nearest',
                             horizontal_flip=True,
                             rescale=1. / 255)
    #For validation
    valid_idg = ImageDataGenerator(rescale=1. /255)

    scores = []
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    n_fold = 1

    k_fold = KFold(n_splits=N_OF_FOLDS, shuffle= True)
    for train, test in k_fold.split(X, y):
        print('------------------------------------------------------------------------')
        print(f'Training fold {n_fold} ...')

        #Create callback to save model for current fold
        mcp_save = ModelCheckpoint(save_dir + get_model_name(i,n_fold),
                                   save_best_only=True,
                                   monitor='loss',
                                   mode='min',
                                   verbose=1)

        model = get_model(i)
        tr = X[train]
        te = y[train]
        history = model.fit(
            X[train],
            y[train],
            batch_size=BATCH_SIZE,
            epochs=N_OF_EPOCHS,
            callbacks=[mcp_save, reduce_lr_loss, early_stopping])

        scores = model.evaluate(X[test], y[test], verbose=0)
        # scores.append({'accuracy':np.average(history.history['accuracy']),
        #                'loss':np.average(history.history['loss'])})

        VALIDATION_ACCURACY.append(scores[1])
        VALIDATION_LOSS.append(scores[0])

        n_fold += 1

    return [VALIDATION_ACCURACY,VALIDATION_LOSS]


plot_scores(k_fold_cross_validation(1))