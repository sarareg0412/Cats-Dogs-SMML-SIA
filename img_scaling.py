import tensorflow as tf
from PIL import Image
import numpy as np

CATS_IMGS = "../CatsDogs/Cats/"
DOGS_IMGS = "../CatsDogs/Dogs/"
SUBDIRS = ["./images/test/", "./images/train/"]
CATS = "cats/"
DOGS = "dogs/"

#MAX_IMGS = 12499
MAX_IMGS = 500
SIZE = 200, 200

photos, labels = list(), list()
#Step one: transform to RGB values and scale cat images
for i in range(MAX_IMGS + 1):
    jpg = Image.open(CATS_IMGS + str(i) + ".jpg")
    rgb = Image.fromarray(np.asarray(jpg), mode="RGB")
    rgb.resize(SIZE, Image.ANTIALIAS)
    rgb.save(SUBDIRS[1] + CATS + str(i) + ".png", )

