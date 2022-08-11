import tensorflow as tf
from PIL import Image

import numpy as np

CATS_IMGS = "../CatsDogs/Cats/"
DOGS_IMGS = "../CatsDogs/Dogs/"
SUBDIRS = ["./images/test/", "./images/train/"]
CATS = "cats/"
DOGS = "dogs/"

# MAX_IMGS = 12499
MAX_IMGS = 10
SIZE = (200, 200)

photos, labels = list(), list()

# Function to convert dataset images to RGB values and resize them
def convert_and_resize(animal_imgs: str, animal: str):
    for i in range(MAX_IMGS + 1):
        jpg = Image.open(animal_imgs + str(i) + ".jpg").convert("RGBA")
        x, y = jpg.size
        rgb = Image.new("RGBA", (x, y), (255, 255, 255))
        rgb.paste(jpg, (0, 0, x, y), jpg)
        rgb.thumbnail(SIZE)
        rgb.save(SUBDIRS[1] + animal + str(i) + ".png", "PNG", quality=100)

# Step one: transform to RGB values and scale cat images
print("> Converting images to RGB values...")
convert_and_resize(CATS_IMGS, CATS)
convert_and_resize(DOGS_IMGS, DOGS)
print("> Images converted and resized.")
