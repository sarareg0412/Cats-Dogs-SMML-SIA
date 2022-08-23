import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from preprocessing import IMGS_PATH, SIZE, BATCH_SIZE


gen = ImageDataGenerator(
    rescale = 1./127.5,
)



train_dataset = gen.flow_from_directory(
    IMGS_PATH,  # Directory where the data is located
    target_size=SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    subset="training",
    seed=123,
    color_mode="rgb"
)

val_dataset = gen.flow_from_directory(
    IMGS_PATH,  # Directory where the data is located
    target_size=SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    subset="validation",
    seed=123,
    color_mode="rgb"
)
