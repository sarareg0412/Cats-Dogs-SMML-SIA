import os
import shutil
import sys

import matplotlib.pyplot as plt


N_OF_FOLDS = 5
EPOCHS = 2


def plot_scores(histories, model_index, save_plot_dir):

    create_dir(save_plot_dir)
    acc, loss, val_acc, val_loss, accs = [], [], [], [], []

    # For each fold there are N_EPOCH values of training loss, validation loss,
    # training accuracy and validation accuracy
    for i in range(N_OF_FOLDS):

        loss = histories[i]['loss']
        acc = histories[i]['accuracy']
        val_loss = histories[i]['val_loss']
        val_acc = histories[i]['val_accuracy']

        plt.figure(figsize=(20, 25))
        x = range(EPOCHS)
        f, axis = plt.subplots(2, sharex=True)

        f.suptitle("Loss and accuracy values")
        axis[0].plot(loss, 'bo--', label='Training Loss')
        axis[0].plot(val_loss, 'ro--', label='Validation Loss')
        axis[0].set_ylabel("Loss value")
        axis[0].legend(loc='upper right')

        axis[1].plot(acc, 'bo--', label='Training Accuracy')
        axis[1].plot(val_acc, 'ro--', label='Validation Accuracy')
        axis[1].set_ylabel("Accuracy value")
        axis[1].set_xlabel("Epoch")
        axis[1].legend(loc='lower right')

        plt.savefig(f"{save_plot_dir}model{model_index}_fold{i}_total_plot.png")

    # accs contains the average of the validation accuracy for each fold
    for i in range(N_OF_FOLDS):
        accs.append(sum(histories[i]['val_accuracy']) / EPOCHS)

    plt.figure(figsize=(12, 8))
    plt.plot(range(N_OF_FOLDS), accs, 'o--', label='Average validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Average accuracy')
    plt.ylabel("Accuracy value")
    plt.xlabel("Epoch")
    plt.text(EPOCHS/2, 3, f'AVG ACCURACY: {sum(accs) / len(accs)}')
    plt.savefig(f"{save_plot_dir}model{model_index}_K_fold_accuracy_plot.png")
    print("Plots correctly saved.")


def plot_losses(losses, save_plot_dir, loss):
    gen_loss, disc_loss = [], []
    for i in range(len(losses)):
        gen_loss.append(losses[i][0])
        disc_loss.append(losses[i][1])

    plt.figure(figsize=(12, 8))
    plt.plot(gen_loss, 'bo--', label='Generator Loss')
    plt.plot(disc_loss, 'ro--', label='Discriminator Loss')
    plt.ylabel("Loss value")
    plt.legend(loc='upper right')

    plt.savefig(f"{save_plot_dir}loss_function_{loss}.png")


# First remove directory, then create it with the passed path
def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)