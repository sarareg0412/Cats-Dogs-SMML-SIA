import glob
import os
import pickle
import warnings

import h5py
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from keras_preprocessing.image import ImageDataGenerator

from tensorflow import keras

import numpy as np


# Function to convert dataset images to RGB values and resize them
# def convert_to_RGB_and_resize(animal_imgs: str, animal: str):
#     for i in range(MAX_IMGS):
#         jpg = Image.open(animal_imgs + str(i) + ".jpg").convert("RGBA")
#         x, y = jpg.size
#         rgb = Image.new("RGBA", (x, y), (255, 255, 255))
#         rgb.paste(jpg, (0, 0, x, y), jpg)
#         rgb = rgb.resize(SIZE)
#         # rgb.save(SUBDIRS[1] + animal + str(i) + ".png", "PNG", quality=100)
#
#
# def convert_to_greyscale_and_resize(animal_imgs: str, animal: str):
#     for i in range(MAX_IMGS + 1):
#         grey = Image.open(animal_imgs + str(i) + ".jpg").convert("LA")
#         grey = grey.resize(SIZE)
#         grey.save(SUBDIRS[1] + animal + str(i) + ".png", "PNG", quality=100)


# def plot_images(photos, labels):
#     ncols, nrows = 4, 8
#     plt.figure(figsize=(ncols * 3, nrows * 3), dpi=90)
#     for i, (img, label) in enumerate(zip(photos, labels)):
#         plt.subplot(nrows, ncols, i + 1)
#         plt.imshow(img.astype(int))
#         assert (label[0] + label[1] == 1.)
#         categ = 'dog' if label > 0.5 else 'cat'
#         plt.title('{} {}'.format(str(label), categ))
#         plt.axis('off')
#
# def plot_img(ds):
#     plt.figure(figsize=(10, 10))
#     for images, labels in ds.take(1):
#         for i in range(9):
#             ax = plt.subplot(3, 3, i + 1)
#             plt.imshow(images[i].numpy().astype("uint8"))
#             plt.title(ds.class_names[labels[i]])
#             plt.axis("off")

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
        print("Images are ok!")
    else:
        print("Removing them...")
        if not os.path.exists(f"{TRASH_PATH}"):
            os.mkdir(f"{TRASH_PATH}")
            os.mkdir(f"{TRASH_PATH}{CATS}")
            os.mkdir(f"{TRASH_PATH}{DOGS}")
        for path in faulty_images:
            print(f"Moving image: {path} to Trash")
            os.rename(f"{path}", f"{TRASH_PATH}{path[len(IMGS_PATH):]}")


IMGS_PATH = "../CatsDogs/"
TRASH_PATH = "../Trash/"
CATS_IMGS_PATH = "../CatsDogs/cat/"
DOGS_IMGS_PATH = "../CatsDogs/dog/"
SUBDIRS = ["./images/test/", "./images/train/"]
CATS = "cat/"
DOGS = "dog/"

# MAX_IMGS = 12499 + 1
MAX_IMGS = 10 + 1
SIZE = (200, 200)

BATCH_SIZE = 30
IMG_HEIGHT = 200
IMG_WIDTH = 200

#rm_faulty_images(IMGS_PATH)

train_ds = tf.keras.utils.image_dataset_from_directory(
    IMGS_PATH,  # Directory where the data is located
    labels="inferred",  # labels are generated from the directory structure
    label_mode="binary",  # labels are encoded as scalars
    batch_size=BATCH_SIZE,  # Size of the batches of data
    validation_split=0.2,  # Fraction of data to reserve for validation
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="rgb"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    IMGS_PATH,
    labels="inferred",
    label_mode="binary",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="rgb"
)

# plot_img(train_ds)
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# # Saving datasets for later training
# train_images = np.concatenate(list(train_ds.map(lambda x, y:x)))
# train_labels = np.concatenate(list(train_ds.map(lambda x, y:y)))
#
# val_images = np.concatenate(list(val_ds.map(lambda x, y:x)))
# val_labels = np.concatenate(list(val_ds.map(lambda x, y:y)))
#
# inputs = np.concatenate((train_images, val_images), axis=0)
# targets = np.concatenate((train_labels, val_labels), axis=0)

# with h5py.File('./input_dataset.hdf5', 'w') as f:
#     f.create_dataset("inputs", data=inputs)
#
# with h5py.File('./target_dataset.hdf5', 'w') as f:
#     f.create_dataset("targets", data=targets)
