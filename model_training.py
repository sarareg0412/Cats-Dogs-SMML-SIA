import os

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow import keras
from preprocessing import train_dataset, BATCH_SIZE, IMGS_PATH, SIZE

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_testing import plot_scores

import pandas as pd
import numpy as np

#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

N_OF_FOLDS = 5
N_OF_EPOCHS = 1
MAX_BATCHES = 25000 / BATCH_SIZE

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

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')


def get_model_name(i):
    return 'model_' + str(i)


def k_fold_cross_validation(model_index):

    TRAINING_LOSSES = []
    VALIDATION_LOSSES = []
    HISTORY = []

    model = get_model(model_index)

    for epoch in range(N_OF_EPOCHS):
        print(f'*****************************EPOCH {epoch}*********************************')

        n_fold = 1
        k_fold = KFold(n_splits=N_OF_FOLDS, shuffle=True)

        # Kfold training loop
        for train, test in k_fold.split(train_dataset):
            print('------------------------------------------------------------------------')
            print(f'Training fold {n_fold} ...')

            # TRAINING LOOP
            print(f"Starting training loop. Max Batches:{MAX_BATCHES}")
            losses = []
            for image_batch in train_dataset:
                if train_dataset.batch_index != 0 and train_dataset.batch_index % 100 == 0:
                    print(f"Round:{train_dataset.batch_index}")

                # Train the model for each batch in the train set of the fold
                if train_dataset.batch_index in train:
                    # This is the equivalent of the model.fit, but done on a series of batches
                    losses = model.train_on_batch(
                        image_batch[0],         # Features
                        image_batch[1],         # Labels
                        #reset_metrics=False,    # Metrics are accumulated across batches
                    )

                if train_dataset.batch_index == 0:
                    print("Training finished.")
                    break

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
                    losses = model.test_on_batch(
                        image_batch[0],
                        image_batch[1],
                        reset_metrics=False,  # Metrics are accumulated across batches
                    )

                if train_dataset.batch_index == 0:
                    print("Validation finished.")
                    break

            print(f"Validation {model.metrics_names} : {losses}")
            VALIDATION_LOSSES.append(losses)
            n_fold += 1

        print(f"Epoch {epoch} done. Saving model")
        model.save(get_model_name(model_index))
        # History contains validation and training loss for each fold, for each epoch
        HISTORY.append([TRAINING_LOSSES, VALIDATION_LOSSES])

    return HISTORY


plot_scores(k_fold_cross_validation(1), 1)
