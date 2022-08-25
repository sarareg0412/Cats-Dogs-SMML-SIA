import os

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

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
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

    TRAINING_HISTORIES = []
    VALIDATION_HISTORIES = []


    model = get_model(model_index)

    for epoch in range(N_OF_EPOCHS):
        print(f'*****************************EPOCH {epoch}*********************************')

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

            # TRAINING LOOP
            i = 0
            print(f"Starting training loop. Max Batches:{MAX_BATCHES}")
            history = []
            for image_batch in train_dataset:
                if i%100 == 0 :
                    print(f"Round:{i}")

                if i > MAX_BATCHES:
                    break
                else:
                    i +=1
                    # Train the model for each batch in the train set of the fold
                    if train_dataset.batch_index in train:

                        history = model.train_on_batch(
                            image_batch[0],
                            image_batch[1],
                            reset_metrics=False,  # Metrics are accumulated across batches
                        )

            print(f"{model.metrics_names}")
            TRAINING_HISTORIES.append(history)

            # VALIDATION LOOP
            i = 0
            print(f"Starting validation loop. Max Batches:{MAX_BATCHES}")
            history = []
            for image_batch in train_dataset:
                if i%100 == 0 :
                    print(f"Round:{i}")

                if i > MAX_BATCHES:
                    break
                else:
                    i +=1
                    # Train the model for each batch in the train set of the fold
                    if train_dataset.batch_index in test:

                        history = model.test_on_batch(
                            image_batch[0],
                            image_batch[1],
                            reset_metrics=False,        # Metrics are accumulated across batches
                        )

            VALIDATION_HISTORIES.append(history)
            n_fold += 1

    return [TRAINING_HISTORIES, VALIDATION_HISTORIES]


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
plot_scores(k_fold_cross_validation(1), 1)

