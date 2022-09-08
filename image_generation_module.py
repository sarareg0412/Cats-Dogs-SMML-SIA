import glob
import os
import time
from IPython import display
import imageio
import tensorflow as tf
from matplotlib import pyplot as plt
from model_testing import create_dir, plot_array
from preprocessing import IMGS_PATH, get_train_and_val_dataset
from keras import layers
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

BATCH_SIZE = 32
MAX_BATCHES = 25000/BATCH_SIZE
tmp_dir = 'tmp/'

IMG_WIDTH = 32
IMG_HEIGHT = 32

N_GEN_IMG = 4
NOISE_SHAPE = 100
SCALE_FACTOR = 4
RESCALING_FACTOR = 127.5

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Used to produce images from a seed.
# The Generator network takes as input a simple random noise N-dimensional vector
# and transforms it according to a learned target distribution.
def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(IMG_WIDTH // SCALE_FACTOR * IMG_WIDTH // SCALE_FACTOR * 256, use_bias=False, input_shape=(NOISE_SHAPE,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 256)))
    assert model.output_shape == (None, IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 256)  # Note: None is the batch size
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    #assert model.output_shape == (None, 28, 28, 1)

    return model


def make_generator_model1():
    model = tf.keras.Sequential()
    model.add(layers.Dense(IMG_WIDTH // SCALE_FACTOR * IMG_WIDTH // SCALE_FACTOR * 128,
                    input_shape=(NOISE_SHAPE,), kernel_initializer=weight_initializer))
    # model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MOMENTUM))
    # model.add(LeakyReLU(alpha=leaky_relu_slope))
    model.add(layers.Reshape((IMG_HEIGHT // SCALE_FACTOR, IMG_WIDTH // SCALE_FACTOR, 128)))

    model = transposed_conv(model, 512, ksize=5, stride_size=1)
    model.add(layers.Dropout(0.4))
    model = transposed_conv(model, 256, ksize=5, stride_size=2)
    model.add(layers.Dropout(0.4))
    model = transposed_conv(model, 128, ksize=5, stride_size=2)
    model = transposed_conv(model, 64, ksize=5, stride_size=2)
    model = transposed_conv(model, 32, ksize=5, stride_size=2)

    model.add(layers.Dense(1, activation='tanh', kernel_initializer=weight_initializer))

    return model


def transposed_conv(model, out_channels, ksize, stride_size, ptype='same'):
    model.add(layers.Conv2DTranspose(out_channels, (ksize, ksize),
                              strides=(stride_size, stride_size), padding=ptype,
                              kernel_initializer=weight_initializer, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
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


def make_discriminator_model1():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', use_bias=False,
                       input_shape=[IMG_HEIGHT, IMG_WIDTH, 1],
                       kernel_initializer=weight_initializer))
    # model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MOMENTUM))
    model.add(layers.LeakyReLU())
    # model.add(Dropout(dropout_rate))

    model = convSN(model, 64, ksize=5, stride_size=2)
    # model = convSN(model, 128, ksize=3, stride_size=1)
    model = convSN(model, 128, ksize=5, stride_size=2)
    # model = convSN(model, 256, ksize=3, stride_size=1)
    model = convSN(model, 256, ksize=5, stride_size=2)
    # model = convSN(model, 512, ksize=3, stride_size=1)
    # model.add(Dropout(dropout_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def convSN(model, out_channels, ksize, stride_size):
    model.add(layers.Conv2D(out_channels, (ksize, ksize), strides=(stride_size, stride_size), padding='same',
                     kernel_initializer=weight_initializer, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(Dropout(dropout_rate))
    return model


# How well the discriminator is able to distinguish real images from fakes
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# How well the generator was able to trick the discriminator
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()

noise = tf.random.normal([1, 100])
#generated_image = generator(noise, training=False)

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
discriminator = make_discriminator_model()
#decision = discriminator(generated_image)
#print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
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


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(N_GEN_IMG, N_GEN_IMG))

    for i in range(predictions.shape[0]):
        plt.subplot(N_GEN_IMG, N_GEN_IMG, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        #plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5), interpolation='nearest')
        plt.axis('off')

    plt.savefig(f'{tmp_dir}image_at_epoch_{epoch:04d}.png')
    #plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        print(f"Starting training for epoch {epoch}. Max batches:{MAX_BATCHES}")
        for image_batch in dataset:
            if dataset.batch_index != 0 and dataset.batch_index % 100 == 0:
                print(f"Round:{dataset.batch_index}")

            # Train the model for each batch in the train set of the fold
            train_step(image_batch[0])
            if dataset.batch_index == 0:
                print(f"Training finished for epoch {epoch}.")
                break

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        print("Saving generated image.")
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            print("Saving model and discriminator losses.")
            checkpoint.save(file_prefix=checkpoint_prefix)
            #plot_array(disc_losses, "Discriminator loss", "disc_losses")

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


anim_file = 'dcgan.gif'

def create_gif():
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(f'{tmp_dir}image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)




train_dataset, val_dataset = get_train_and_val_dataset(rescale=RESCALING_FACTOR,
                                                        size = (IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        validation=0.2)

# Could add a Data Augmentation step

weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.2, mean=0.0, seed=42)
create_dir(tmp_dir)
print("Starting training.")
train(train_dataset, EPOCHS)
print("Training completed.")
print("Creating GIF of saved images.")
create_gif()
print("Done.")



