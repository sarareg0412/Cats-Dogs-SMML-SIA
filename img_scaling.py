import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
from keras_preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

CATS_IMGS = "../CatsDogs/Cats/"
DOGS_IMGS = "../CatsDogs/Dogs/"
SUBDIRS = ["./images/test/", "./images/train/"]
CATS = "cats/"
DOGS = "dogs/"

# MAX_IMGS = 12499 + 1
MAX_IMGS = 10 + 1
SIZE = (200, 200)

# List of rgb values of the pictures, and list of the lables for each pic (0 cat, 1 dog)
photos, labels = list(), list()


# Function to convert dataset images to RGB values and resize them
def convert_to_RGB_and_resize(animal_imgs: str, animal: str):
    for i in range(MAX_IMGS):
        jpg = Image.open(animal_imgs + str(i) + ".jpg").convert("RGBA")
        x, y = jpg.size
        rgb = Image.new("RGBA", (x, y), (255, 255, 255))
        rgb.paste(jpg, (0, 0, x, y), jpg)
        rgb = rgb.resize(SIZE)
        photos.append(img_to_array(rgb))
        if animal == CATS:
            labels.append(0.0)
        else:
            labels.append(1.0)
        # rgb.save(SUBDIRS[1] + animal + str(i) + ".png", "PNG", quality=100)


def convert_to_greyscale_and_resize(animal_imgs: str, animal: str):
    for i in range(MAX_IMGS + 1):
        grey = Image.open(animal_imgs + str(i) + ".jpg").convert("LA")
        grey = grey.resize(SIZE)
        grey.save(SUBDIRS[1] + animal + str(i) + ".png", "PNG", quality=100)

def plot_images(photos, labels):
    ncols, nrows = 4, 8
    fig = plt.figure(figsize=(ncols * 3, nrows * 3), dpi=90)
    for i, (img, label) in enumerate(zip(photos, labels)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img.astype(int))
        assert (label[0] + label[1] == 1.)
        categ = 'dog' if label > 0.5 else 'cat'
        plt.title('{} {}'.format(str(label), categ))
        plt.axis('off')

# Step one: transform to RGB values and scale cat images
print("> Converting images to RGB values...")
convert_to_RGB_and_resize(CATS_IMGS, CATS)
convert_to_RGB_and_resize(DOGS_IMGS, DOGS)
print("> Images converted and resized.")
photos = np.asarray(photos)
labels = np.asarray(labels)
np.save('cats_and_dogs_rgb_values.npy', photos)
np.save('cats_and_dogs_labels.npy', labels)
plot_images(photos, labels)

imgdatagen = ImageDataGenerator(
    validation_split= 0.2   #Use 20% of the images for validation, 80% for training
)

batch_size = 30

train_dataset = imgdatagen.flow_from_directory(

)

