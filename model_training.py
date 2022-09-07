import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from preprocessing import get_train_and_val_dataset

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_testing import plot_scores
import torch

#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

BATCH_SIZE = 64
N_OF_FOLDS = 5
N_OF_EPOCHS = 5
MAX_BATCHES = 25000 / BATCH_SIZE
CHANNELS = 1
IMG_HEIGHT = 128
IMG_WIDTH = 128

def get_model(i: int):
    if i == 1:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))
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


reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=15, verbose=0, mode='min')


def get_model_name(i):
    return 'model_' + str(i)


def k_fold_cross_validation(model_index):

    TRAINING_LOSSES = []
    VALIDATION_LOSSES = []
    HISTORY = []

    model = get_model(model_index)

    n_fold = 1
    k_fold = KFold(n_splits=N_OF_FOLDS, shuffle=True)

    # Kfold training loop
    for train, test in k_fold.split(train_dataset):
        indices_train = torch.tensor(train)
        indices_val = torch.tensor(test)
        X_train, y_train, X_test, y_test = [], [], [], []

        for image_batch in train_dataset:
            if train_dataset.batch_index % 100 == 0:
                print(f"Round:{train_dataset.batch_index}")

            if train_dataset.batch_index in train:
                X_train.append(image_batch[0])
                y_train.append(image_batch[1])
            else:
                X_test.append(image_batch[0])
                y_test.append(image_batch[1])

            if train_dataset.batch_index == 0:
                print(f'Folds generation finished')
                break


        # TRAINING LOOP
        print('------------------------------------------------------------------------')
        print(f'Training fold {n_fold} ...')
        print(f"Starting training loop. Max Batches:{MAX_BATCHES}")
        losses = []

        # Train the model for each batch in the train set of the fold
        losses = model.fit(
            torch.tensor(X_train),         # Features
            torch.tensor(y_train),         # Labels
            verbose=1,
            epochs= N_OF_EPOCHS,
            callbacks=[reduce_lr_loss, early_stopping],
        )

        print(f"Training {model.metrics_names} : {losses}")
        TRAINING_LOSSES.append(losses)

        # VALIDATION LOOP
        print(f"Starting validation loop. Max Batches:{MAX_BATCHES}")
        losses = []
        for image_batch in train_dataset:

            if train_dataset.batch_index % 100 == 0:
                print(f"Round:{train_dataset.batch_index}")

            # Train the model for each batch in the train set of the fold
            if train_dataset.batch_index in test:
                #losses = model.test_on_batch(
                losses = model.evaluate(
                    image_batch[0],
                    image_batch[1],
                    verbose=1,
                    callbacks=[reduce_lr_loss, early_stopping],
                )

            if train_dataset.batch_index == 0:
                print("Validation finished.")
                break

        print(f"Validation {model.metrics_names} : {losses}")
        VALIDATION_LOSSES.append(losses)
        n_fold += 1

        model.save(get_model_name(model_index))
        # History contains validation and training loss for each fold, for each epoch
        HISTORY.append([TRAINING_LOSSES, VALIDATION_LOSSES])

    return HISTORY


train_dataset, val_dataset = get_train_and_val_dataset( rescale=255.,
                                                        size = (IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        validation=0.0)
plot_scores(k_fold_cross_validation(1), 1)
