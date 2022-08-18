import h5py
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow import keras
from img_scaling import train_ds, val_ds, BATCH_SIZE, IMGS_PATH

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import sklearn.model_selection as sklrn
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


mcp_save = ModelCheckpoint('/tmp/ckeckpoint', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')

# history = get_model(1).fit(
#     train_ds,
#     epochs=3,
#     validation_data=val_ds,
#     callbacks=[mcp_save, reduce_lr_loss, early_stopping],
# )

N_OF_FOLDS = 5
N_OF_EPOCHS = 5
SHUFFLE_BUFFER_SIZE = 100


def k_fold_cross_validation(model, n_folds=N_OF_FOLDS, epochs=N_OF_EPOCHS, batch_size=BATCH_SIZE):
    # dictionary where data will be stored and counter
    dic = {}
    i = 0.0

    ds = train_ds.concatenate(val_ds)

    lookup_images = {}
    lookup_labels = {}
    for i, (x, y) in enumerate(ds):
        lookup_images[i] = x
        lookup_labels[i] = y

    folds = KFold(n_folds, shuffle=True).split(np.arange(len(ds)))

    # for train_index, val_index in folds:
    #     x_train_kf, x_val_kf = train_ds[train_index], train_ds[val_index]
    #     y_train_kf, y_val_kf = val_ds[train_index], val_ds[val_index]
    for train, test in folds:

        images_train = np.concatenate(list(map(lookup_images.get, train)))
        labels_train = np.concatenate(list(map(lookup_labels.get, train)))

        images_test = np.concatenate(list(map(lookup_images.get, test)))
        labels_test = np.concatenate(list(map(lookup_labels.get, test)))

        history = model.fit(
            images_train,
            labels_train,
            callbacks=[mcp_save, reduce_lr_loss, early_stopping],
            epochs=epochs
        )

        if dic == {}:
            # if dictionary is empty, values will be put there
            dic['train_acc'] = np.array(history.history['accuracy'])
            dic['train_loss'] = np.array(history.history['loss'])
            dic['val_acc'] = np.array(history.history['val_accuracy'])
            dic['val_loss'] = np.array(history.history['val_loss'])
        else:
            # if dictionary is not empty, values will be added element wise
            dic['train_acc'] += np.array(history.history['accuracy'])
            dic['train_loss'] += np.array(history.history['loss'])
            dic['val_acc'] += np.array(history.history['val_accuracy'])
            dic['val_loss'] += np.array(history.history['val_loss'])

        i += 1

    for k in dic:
        # each number in each array in the dictionary will be divided by the number of iterations, producing the mean of all the values read
        dic[k] /= i

    return dic

def save_dict():
    dict = k_fold_cross_validation(model=get_model(1))

    with h5py.File('dict_model1.h5', 'w') as hf:
        hf.create_dataset("dict_model1",  data=dict)


save_dict()