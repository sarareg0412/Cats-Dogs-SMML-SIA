import os
import time
from IPython import display
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import create_dir, add_value_to_avg, remove_if_exists
from preprocessing import get_train_and_val_dataset_IDG

from keras.layers import Conv2D, \
    Dropout, Flatten, Dense, BatchNormalization, ReLU, Reshape, Conv2DTranspose, LeakyReLU
from tensorflow.python.ops.numpy_ops import np_config
from utils import plot_losses

tf.config.run_functions_eagerly(True)
np_config.enable_numpy_behavior()

BATCH_SIZE = 128
MAX_BATCHES = 25000 / BATCH_SIZE
img_gen_dir = 'img_gen/'
LOSS_INDEX = 0  # 0:BCE, 1:MSE

IMG_WIDTH = 28
IMG_HEIGHT = 28

N_GEN_IMG = 4
NOISE_SHAPE = 100
SCALE_FACTOR = 4
RESCALING_FACTOR = 127.5

EPOCHS = 100
SAVE_IMAGES_INTERVAL = 5
SAVE_PLOT_INTERVAL = 10
noise_dim = 100
num_examples_to_generate = 16


def get_loss_function(par):
    if par == 0:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        return tf.keras.losses.MeanSquaredError()


# Used to produce images from a seed.
# The Generator network takes as input a simple random noise N-dimensional vector
# and transforms it according to a learned target distribution.
def make_generator_model():
    model = tf.keras.Sequential()

    model.add(Dense(IMG_WIDTH // SCALE_FACTOR * IMG_WIDTH // SCALE_FACTOR * 256, use_bias=False,
                    input_shape=(NOISE_SHAPE,)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Reshape((IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 256)))
    model.add(Dropout(0.4))

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.4))

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


# CNN-based image classifier
# The discriminator outputs a probability that the input image is real or fake [0, 1].
def make_discriminator_model(loss_index):
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_WIDTH, IMG_HEIGHT, 1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    if loss_index:
        model.add(LeakyReLU())

    return model


# How well the discriminator is able to distinguish real images from fakes
def discriminator_loss(real_output, fake_output):
    # The first argument is the array of true labels, the second one is the predicted one
    real_loss = loss(tf.ones_like(real_output), real_output)
    fake_loss = loss(tf.zeros_like(fake_output), fake_output)
    return real_loss, fake_loss


# How well the generator was able to trick the discriminator
def generator_loss(fake_output):
    return loss(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        r_disc_loss, f_disc_loss = discriminator_loss(real_output, fake_output)
        disc_loss = r_disc_loss + f_disc_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss.numpy(), r_disc_loss.numpy(), f_disc_loss.numpy()


def generate_and_save_images(model, loss_name, epoch, test_input):
    create_dir(f'{img_gen_dir}{loss_name}/')
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False).numpy()

    fig = plt.figure(figsize=(N_GEN_IMG, N_GEN_IMG))

    for i in range(predictions.shape[0]):
        plt.subplot(N_GEN_IMG, N_GEN_IMG, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    name_fig = f'{img_gen_dir}{loss_name}/image_at_epoch_{epoch:04d}.png'
    remove_if_exists(name_fig)
    plt.savefig(name_fig)


def train(dataset, epochs):
    i = 0
    name = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(name)
    if tf.train.latest_checkpoint(checkpoint_dir):
        print(f"Restored from {name}")
        # Ex. Latest checkpoint was 10 (at epoch 100), now training
        # will start from 101
        i = int(f'{name[-2]}{name[-1]}') * 10
    else:
        print("Initializing from scratch.")

    epoch_losses = []
    for epoch in range(i, epochs):
        print(f"Starting training for epoch {epoch + 1}/{EPOCHS}. Max batches:{MAX_BATCHES}")
        losses = [0.0, 0.0, 0.0]
        start = time.time()
        for image_batch in dataset:
            if dataset.batch_index == 0:
                print('Training finished. Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                break
            # Train the model for each batch in the train set
            gen_l, r_disc_l, f_disc_l = train_step(image_batch[0])
            losses = [add_value_to_avg(losses[0], gen_l, dataset.batch_index),
                      add_value_to_avg(losses[1], r_disc_l, dataset.batch_index),
                      add_value_to_avg(losses[2], f_disc_l, dataset.batch_index)]

        epoch_losses.append(losses)
        if epoch == 0 or (epoch + 1) % SAVE_IMAGES_INTERVAL == 0:
            print("Saving generated images")
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     loss_name,
                                     epoch + 1,
                                     seed)

        # Produce plot and save the model
        if (epoch + 1) % SAVE_PLOT_INTERVAL == 0:
            print("Saving model and plotting discriminator losses.")
            checkpoint.save(file_prefix=checkpoint_prefix)
            plot_losses(epoch_losses, img_gen_dir, loss_name, epoch + 1, BATCH_SIZE)


# Could add a Data Augmentation step
def start():
    create_dir(img_gen_dir)
    create_dir(img_gen_dir + loss_name)
    print(f"Starting training with loss {loss_name}.")
    train(train_dataset, EPOCHS)
    print("Training completed.")
    print("Done.")


generator = make_generator_model()
noise = tf.random.normal([1, 100])

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.2, mean=0.0, seed=42)

train_dataset, val_dataset = get_train_and_val_dataset_IDG(rescale=RESCALING_FACTOR,
                                                           size=(IMG_WIDTH, IMG_HEIGHT),
                                                           batch_size=BATCH_SIZE,
                                                           validation=0.0)

loss = get_loss_function(LOSS_INDEX)
discriminator = make_discriminator_model(LOSS_INDEX)
loss_name = 'mse' if LOSS_INDEX else 'bce'
checkpoint_dir = img_gen_dir + loss_name + '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
start()
