import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from preprocessing import get_train_and_val_dataset, get_train_and_val_dataset_IDG

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization, AveragePooling2D, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_testing import plot_scores

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

BATCH_SIZE = 32
N_OF_FOLDS = 5
N_OF_EPOCHS = 2
MAX_BATCHES = 25000 / BATCH_SIZE
CHANNELS = 1
IMG_HEIGHT = 50
IMG_WIDTH = 50


# Returns the model chosen based on the index
def get_model(i: int):
    if i == 1:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name='conv0'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool0'))

        model.add(Conv2D(32, (3, 3), name='conv1'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool1'))

        model.add(Conv2D(32, (3, 3), name='conv2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool2'))

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

    if i == 2:
        model = Sequential()

        model.add(Conv2D(32, (5, 5), strides=(1, 1), name='conv0', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))

        model.add(BatchNormalization(axis=3, name='bn0'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D((2, 2), name='max_pool'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
        model.add(Activation('relu'))
        model.add(AveragePooling2D((3, 3), name='avg_pool'))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(300, activation="relu", name='rl'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid', name='sm'))
        adam_optimizer = tf.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=adam_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # model.summary()
        return model


reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')


def k_fold_cross_validation(model_index):
    HISTORY = []
    model = get_model(model_index)
    n_fold = 1
    k_fold = KFold(n_splits=N_OF_FOLDS, shuffle=True)

    # Kfold training loop
    for train, test in k_fold.split(train_dataset):
        print(f'---------------------------- FOLD {n_fold} ----------------------------')
        X_train, y_train, X_test, y_test = None, None, None, None

        print(f"Starting folded datasets generation. Max Batches:{MAX_BATCHES}")
        for image_batch in train_dataset:
            if train_dataset.batch_index == 0:
                print(f'Training and validation dataset generation for fold {n_fold} finished.')
                break

            if train_dataset.batch_index % 100 == 0:
                print(f"Round:{train_dataset.batch_index}")

            if train_dataset.batch_index in train:
                if X_train is None:
                    X_train = np.array(image_batch[0])
                    y_train = np.array(image_batch[1])
                else:
                    X_train = np.insert(X_train, 1, np.array(image_batch[0]), axis=0)
                    y_train = np.insert(y_train, 1, np.array(image_batch[1]), axis=0)
            else:
                if X_test is None:
                    X_test = np.array(image_batch[0])
                    y_test = np.array(image_batch[1])
                else:
                    X_test = np.insert(X_test, 1, np.array(image_batch[0]), axis=0)
                    y_test = np.insert(y_test, 1, np.array(image_batch[1]), axis=0)

        # TRAINING AND VALIDATION LOOP
        print(f'Starting training for fold {n_fold} ...')

        # Train the model for each batch in the train set of the fold
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            batch_size=BATCH_SIZE,
            verbose=1,
            epochs=N_OF_EPOCHS,
            callbacks=[reduce_lr_loss, early_stopping, mcp_save],
        )

        print(f"Training {model.metrics_names} : {history.history}")
        HISTORY.append(history.history)
        n_fold += 1

    return HISTORY


train_dataset, val_dataset = get_train_and_val_dataset_IDG(rescale=255.,
                                                           size=(IMG_WIDTH, IMG_HEIGHT),
                                                           batch_size=BATCH_SIZE,
                                                           validation=0.0)
plot_scores(k_fold_cross_validation(1), 1)
