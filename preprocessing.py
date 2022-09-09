import os
import glob
import re
import shutil
import warnings
from builtins import float

import h5py
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator

IMGS_PATH = "../CatsDogs/"
TRASH_PATH = "../Trash/"
CATS_IMGS_PATH = "../CatsDogs/Cats/"
DOGS_IMGS_PATH = "../CatsDogs/Dogs/"
SUBDIRS = ["./images/test/", "./images/train/"]
CATS = "Cats"
DOGS = "Dogs"
SIZE = (128, 128)

BATCH_SIZE = 64
# matches any string with the substring ".<digits>."
# such as dog.666.jpg
pattern = re.compile(r'.*\.(\d+)\..*')


def trash_path(dirname):
    '''return the path of the Trash directory,
    where the bad dog and bad cat images will be moved.
    Note that the Trash directory should not be within the dogs/
    or the cats/ directory, or Keras will still find these pictures.
    '''
    return os.path.join(TRASH_PATH, dirname)


def cleanup(ids, dirname):
    '''move away images with these ids in dirname
    '''
    # if it exists, it is first removed
    trash = trash_path(dirname)
    if os.path.isdir(trash):
        shutil.rmtree(trash)
    os.makedirs(trash, exist_ok=True)

    for fname in ids:
        print('moving to {}: {}'.format(trash, fname))
        shutil.move(fname, trash)


def restore(dirname):
    '''restores files in the trash
    I will need this to restore this tutorial to initial state for you
    and you might need it if you want to try training the network
    without the cleaning of bad images
    '''
    # os.chdir(IMGS_PATH)
    # oldpwd = os.getcwd()
    # os.chdir(dirname)
    trash = trash_path(dirname)
    # print(trash)
    for fname in os.listdir(trash):
        trash_name = os.path.join(trash, fname)
        print('restoring', trash_name)
        shutil.move(trash_name, IMGS_PATH + dirname + "/" + fname)
    # os.chdir(oldpwd)


def rm_faulty_images(path: str = None):
    warnings.filterwarnings(
        "ignore",
        "(Possibly )?corrupt EXIF data",
        UserWarning
    )

    warnings.filterwarnings(
        "ignore",
        "extraneous bytes before marker",
        UserWarning
    )

    img_paths = glob.glob(os.path.join(path, '*/*.*'))
    faulty_images = []

    print("Bad paths:")
    for image_path in img_paths:
        try:
            # we deal also with PIL for torch
            # (tf skips jpgs that causes training issues)
            img_bytes = tf.io.read_file(image_path)
            _ = tf.image.decode_image(img_bytes)
            _ = Image.open(image_path)
        except (
                tf.errors.InvalidArgumentError,
                UnidentifiedImageError
        ) as e:
            print(f"- Found bad path at: {image_path} - {e}")
            faulty_images.append(image_path)

    if len(faulty_images) == 0:
        print("No faulty  images found.")
    else:
        print("Removing them...")
        if not os.path.exists(f"{TRASH_PATH}"):
            os.mkdir(f"{TRASH_PATH}")
            os.mkdir(f"{TRASH_PATH}{CATS}")
            os.mkdir(f"{TRASH_PATH}{DOGS}")
        for path in faulty_images:
            print(f"Moving image: {path} to Trash")
            shutil.move(f"{path}", f"{TRASH_PATH}{path[len(IMGS_PATH):]}")


# restore(CATS)
# restore(DOGS)
# rm_faulty_images(IMGS_PATH)

def get_train_and_val_dataset(rescale: float, size: (int, int), batch_size: int):
    normalization_layer = tf.keras.layers.Rescaling(1. / rescale)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        IMGS_PATH,                  # Directory where the data is located
        labels='inferred',
        label_mode='binary',
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=size,
        seed=123,
        interpolation='nearest'
    )

    normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))

    return normalized_train_ds


def get_train_and_val_dataset_IDG(rescale: float, size: (int, int), batch_size: int, validation: float):
    gen = ImageDataGenerator(
        rescale=1 / rescale,
        validation_split=validation
    )

    train_dataset = gen.flow_from_directory(
        IMGS_PATH,  # Directory where the data is located
        target_size=size,
        class_mode='binary',
        batch_size=batch_size,
        subset="training",
        seed=123,
        color_mode="grayscale"
    )

    val_dataset = gen.flow_from_directory(
        IMGS_PATH,  # Directory where the data is located
        target_size=size,
        class_mode='binary',
        batch_size=batch_size,
        subset="validation",
        seed=123,
        color_mode="grayscale"
    )

    return train_dataset, val_dataset
