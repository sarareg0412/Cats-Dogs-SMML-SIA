import glob
import os
import time
from IPython import display
import imageio
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import create_dir, add_value_to_avg
from preprocessing import get_train_and_val_dataset_IDG
from keras import layers
from tensorflow.python.ops.numpy_ops import np_config
from utils import plot_losses

tf.config.run_functions_eagerly(True)
np_config.enable_numpy_behavior()

BATCH_SIZE = 32
MAX_BATCHES = 25000 / BATCH_SIZE
img_gen_dir = 'img_gen/'
tmp_dir = 'tmp/'

IMG_WIDTH = 32
IMG_HEIGHT = 32

N_GEN_IMG = 4
NOISE_SHAPE = 100
SCALE_FACTOR = 4
RESCALING_FACTOR = 127.5

EPOCHS = 2
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.2, mean=0.0, seed=42)


def get_loss_function(par):
    if par == 0:
        # This method returns a helper function to compute cross entropy loss
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        return tf.keras.losses.MeanSquaredError()



# Used to produce images from a seed.
# The Generator network takes as input a simple random noise N-dimensional vector
# and transforms it according to a learned target distribution.
def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(IMG_WIDTH // SCALE_FACTOR * IMG_WIDTH // SCALE_FACTOR * 256, use_bias=False,
                           input_shape=(NOISE_SHAPE,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 256)))
    #assert model.output_shape == (None, IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 256)  # Note: None is the batch size
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    #assert model.output_shape == (None, IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 28, 28, 1)

    return model


# CNN-based image classifier
# The discriminator outputs a probability that the input image is real or fake [0, 1].
def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[IMG_HEIGHT, IMG_WIDTH, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# How well the discriminator is able to distinguish real images from fakes
def discriminator_loss(real_output, fake_output):
    real_loss = loss(tf.ones_like(real_output), real_output)
    fake_loss = loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# How well the generator was able to trick the discriminator
def generator_loss(fake_output):
    return loss(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()

noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)

# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
discriminator = make_discriminator_model()
# decision = discriminator(generated_image)
# print (decision)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = '/training_checkpoints'
checkpoint_prefix = os.path.join(img_gen_dir + checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


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
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss.numpy(), disc_loss.numpy()


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(N_GEN_IMG, N_GEN_IMG))

    for i in range(predictions.shape[0]):
        plt.subplot(N_GEN_IMG, N_GEN_IMG, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        # plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5), interpolation='nearest')
        plt.axis('off')

    plt.savefig(f'{img_gen_dir}{tmp_dir}image_at_epoch_{epoch:04d}.png')
    # plt.show()


def train(dataset, epochs, loss_name):
    losses, epoch_losses = (0.0, 0.0), []
    for epoch in range(epochs):
        print(f"Starting training for epoch {epoch+1}/{EPOCHS}. Max batches:{MAX_BATCHES}")

        start = time.time()
        for image_batch in dataset:
            if dataset.batch_index == 0:
                print('Training finished. Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                break
            # Train the model for each batch in the train set of the fold and save
            # generator and discriminator losses
            gen_l, disc_l = train_step(image_batch[0])
            losses = (add_value_to_avg(losses[0],gen_l,dataset.batch_index),
                      add_value_to_avg(losses[1], gen_l, dataset.batch_index))

        epoch_losses.append(losses)
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            print("Saving model and discriminator losses.")
            checkpoint.save(file_prefix=checkpoint_prefix)
            # plot_array(disc_losses, "Discriminator loss", "disc_losses")

    # Generate after the final epoch
    display.clear_output(wait=True)

    plot_losses(epoch_losses, img_gen_dir, loss_name)

    generate_and_save_images(generator,
                             epochs,
                             seed)


def create_gif(loss_name):
    anim_file = f'dcgan_{loss_name}.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(f'{img_gen_dir}{tmp_dir}image*.png')
        filenames = sorted(filenames)

        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)


train_dataset, val_dataset = get_train_and_val_dataset_IDG(rescale=RESCALING_FACTOR,
                                                           size=(IMG_WIDTH, IMG_HEIGHT),
                                                           batch_size=BATCH_SIZE,
                                                           validation=0.0)


# Could add a Data Augmentation step
def start(loss_index):
    if loss_index == 0:
        loss_name = "bce"
    else:
        loss_name = "mse"

    create_dir(img_gen_dir)
    create_dir(img_gen_dir + tmp_dir)
    print("Starting training.")
    train(train_dataset, EPOCHS, loss_name)
    print("Training completed.")
    print("Creating GIF of saved images.")
    create_gif(loss_name)
    print("Done.")


loss = get_loss_function(0)
start(0)

loss = get_loss_function(1)
start(1)


