import os
import glob
import re
import shutil
from builtins import float

import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator

from utils import create_dir

IMGS_PATH = "../CatsDogs/"
TRASH_PATH = "../Trash/"
CATS_IMGS_PATH = "../CatsDogs/Cats/"
DOGS_IMGS_PATH = "../CatsDogs/Dogs/"
SUBDIRS = ["./images/test/", "./images/train/"]
CATS = "Cats"
DOGS = "Dogs"

pattern = re.compile(r'.*\.(\d+)\..*')

def trash_path(dirname):
    '''return the path of the Trash directory'''
    return os.path.join(TRASH_PATH, dirname)


def cleanup(ids, dirname):
    # if it exists, it is first removed
    trash = trash_path(dirname)
    if os.path.isdir(trash):
        shutil.rmtree(trash)
    os.makedirs(trash, exist_ok=True)

    for fname in ids:
        print('moving to {}: {}'.format(trash, fname))
        shutil.move(fname, trash)


def restore(dirname):

    '''restores files from trash'''
    trash = trash_path(dirname)
    for fname in os.listdir(trash):
        trash_name = os.path.join(trash, fname)
        print(f'Restoring {trash_name}')
        shutil.move(trash_name, IMGS_PATH + dirname + "/" + fname)


def remove_corrupted_images(path: str = None):
    img_paths = glob.glob(os.path.join(path, '*/*.*'))
    bad_images = []

    print("Corrupted images:")
    for image_path in img_paths:
        try:
            img_bytes = tf.io.read_file(image_path)
            tf.image.decode_image(img_bytes)
            Image.open(image_path)
        except (
                tf.errors.InvalidArgumentError,
                UnidentifiedImageError
        ) as e:
            print(f"Found corrupted image at: {image_path} - {e}")
            bad_images.append(image_path)

    if len(bad_images) == 0:
        print("No corrupted images found.")
    else:
        print("Removing them...")
        create_dir(TRASH_PATH)
        create_dir(TRASH_PATH + CATS)
        create_dir(TRASH_PATH + DOGS)
        for path in bad_images:
            print(f"Moving image: {path} to Trash")
            shutil.move(f"{path}", f"{TRASH_PATH}{path[len(IMGS_PATH):]}")


#restore(CATS)
#restore(DOGS)
#remove_corrupted_images(IMGS_PATH)

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
