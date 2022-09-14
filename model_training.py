import numpy as np
import tensorflow as tf
import torch.nn
from keras import Model, Input
from keras.applications import vgg16
from keras.applications.resnet import ResNet, ResNet50
from sklearn.model_selection import KFold
from preprocessing import get_train_and_val_dataset, get_train_and_val_dataset_IDG

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, \
    Dropout, Flatten, Dense, Activation, \
    BatchNormalization, AveragePooling2D, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils import plot_scores, create_dir

import timm
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

BATCH_SIZE = 32
N_OF_FOLDS = 5
N_OF_EPOCHS = 2
MAX_BATCHES = 25000 / BATCH_SIZE
CHANNELS = 1
IMG_HEIGHT = 50
IMG_WIDTH = 50
bin_class_dir = 'bin_class/'
save_plot_dir = "plots/"


# Returns the model chosen based on the index
def get_model(i: int):
    if i == 1:
        # This model has a single, fully connected hidden layer
        model = Sequential()
        # Specify the shape of the input
        model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), name="input_layer"))
        model.add(Flatten(name='flatten_hidden_Layer'))
        model.add(Dense(1024, activation='relu', name='hidden_Layer'))
        model.add(Dense(1, activation='sigmoid', name='output_Layer'))

        adam_optimizer = tf.optimizers.Adam(learning_rate=0.001)

        model.compile(
            optimizer=adam_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model

    if i == 2:
        #Convolutional NN
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

        model.summary()
        return model

    if i == 3:

        vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        custom_model = Sequential()

        for layer in vgg16_model.layers[:-1]:
            custom_model.add(layer)

        #Last layer for classification
        custom_model.add(Flatten(name='last_flatten'))
        custom_model.add(Dense(512, activation='relu'))
        custom_model.add(Dense(1))
        custom_model.add(Activation('softmax'))

        adam_optimizer = tf.optimizers.Adam(learning_rate=0.001)

        custom_model.compile(
            optimizer=adam_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        custom_model.summary()
        return custom_model


reduce_lr_acc = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=0, mode='max')


def get_model_name(i):
    return 'model_'+str(i)+'.hdf5'


def k_fold_cross_validation(model_index):
    HISTORY = []
    model = get_model(model_index)
    n_fold = 1
    k_fold = KFold(n_splits=N_OF_FOLDS, shuffle=True)
    # Create callback to save model for current fold
    mcp_save = ModelCheckpoint(bin_class_dir + get_model_name(model_index),
                               save_best_only=True,
                               monitor='val_accuracy',
                               mode='max',
                               verbose=1)
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

        #Necessary for VG16 models only work with images with 3 channels
        #while grayscale only have 1
        if model_index == 3:
            X_test = X_test.repeat(3,axis=-1)
            X_train = X_train.repeat(3,axis=-1)

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
            callbacks=[reduce_lr_acc, early_stopping, mcp_save],
        )

        print(f"Training {model.metrics_names} : {history.history}")
        HISTORY.append(history.history)
        n_fold += 1

    return HISTORY


train_dataset, val_dataset = get_train_and_val_dataset_IDG(rescale=255.,
                                                           size=(IMG_WIDTH, IMG_HEIGHT),
                                                           batch_size=BATCH_SIZE,
                                                           validation=0.0)
create_dir(bin_class_dir)
plot_scores(k_fold_cross_validation(1), 1, bin_class_dir + save_plot_dir, N_OF_EPOCHS)
