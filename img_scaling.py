import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow import keras

import numpy as np

# Function to convert dataset images to RGB values and resize them
def convert_to_RGB_and_resize(animal_imgs: str, animal: str):
    for i in range(MAX_IMGS):
        jpg = Image.open(animal_imgs + str(i) + ".jpg").convert("RGBA")
        x, y = jpg.size
        rgb = Image.new("RGBA", (x, y), (255, 255, 255))
        rgb.paste(jpg, (0, 0, x, y), jpg)
        rgb = rgb.resize(SIZE)
        # rgb.save(SUBDIRS[1] + animal + str(i) + ".png", "PNG", quality=100)


def convert_to_greyscale_and_resize(animal_imgs: str, animal: str):
    for i in range(MAX_IMGS + 1):
        grey = Image.open(animal_imgs + str(i) + ".jpg").convert("LA")
        grey = grey.resize(SIZE)
        grey.save(SUBDIRS[1] + animal + str(i) + ".png", "PNG", quality=100)


def plot_images(photos, labels):
    ncols, nrows = 4, 8
    plt.figure(figsize=(ncols * 3, nrows * 3), dpi=90)
    for i, (img, label) in enumerate(zip(photos, labels)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img.astype(int))
        assert (label[0] + label[1] == 1.)
        categ = 'dog' if label > 0.5 else 'cat'
        plt.title('{} {}'.format(str(label), categ))
        plt.axis('off')

def plot_img(ds):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(ds.class_names[labels[i]])
            plt.axis("off")


IMGS_PATH = "../CatsDogs/"
CATS_IMGS_PATH = "../CatsDogs/cat/"
DOGS_IMGS_PATH = "../CatsDogs/dog/"
SUBDIRS = ["./images/test/", "./images/train/"]
CATS = "cat/"
DOGS = "dog/"

# MAX_IMGS = 12499 + 1
MAX_IMGS = 10 + 1
SIZE = (200, 200)

batch_size = 30
img_height = 200
img_width = 200

train_ds = tf.keras.utils.image_dataset_from_directory(
  IMGS_PATH,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode="int",
  labels="inferred",
  color_mode="rgb"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  IMGS_PATH,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode="int",
  labels="inferred",
  color_mode="rgb"
)

#plot_img(train_ds)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
