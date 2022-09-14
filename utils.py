import os
import shutil
import sys

import matplotlib.pyplot as plt


N_OF_FOLDS = 5


def plot_scores(histories, model_index, save_plot_dir, epochs):

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
        f, axis = plt.subplots(2, sharex=True)

        f.suptitle("Loss and accuracy values")
        axis[0].plot(range(1, epochs + 1), loss, 'bo--', label='Training Loss')
        axis[0].plot(range(1, epochs + 1), val_loss, 'ro--', label='Validation Loss')
        axis[0].set_ylabel("Loss value")
        axis[0].legend(loc='upper right')

        axis[1].plot(range(1, epochs + 1), acc, 'bo--', label='Training Accuracy')
        axis[1].plot(range(1, epochs + 1), val_acc, 'ro--', label='Validation Accuracy')
        axis[1].set_ylabel("Accuracy value")
        axis[1].set_xlabel("Epoch")
        axis[1].legend(loc='lower right')

        plt.savefig(f"{save_plot_dir}model{model_index}_fold{i}_total_plot.png")

    # accs contains the average of the validation accuracy for each fold
    for i in range(N_OF_FOLDS):
        accs.append(sum(histories[i]['val_accuracy']) / epochs)

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, N_OF_FOLDS + 1), accs, 'o--')
    plt.legend(loc='lower right')
    plt.title('Average validation accuracy')
    plt.ylabel("Accuracy value")
    plt.xlabel("Fold")
    plt.text(3, min(accs), 'AVG ACCURACY: {:.4f}'.format(sum(accs) / len(accs)))

    for x, y in zip(range(1, N_OF_FOLDS + 1), accs):
        label = "{:.4f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.savefig(f"{save_plot_dir}model{model_index}_K_fold_accuracy_plot.png")
    print("Plots correctly saved.")


def plot_losses(losses, save_plot_dir, loss):
    gen_loss, disc_loss = [], []
    for i in range(len(losses)):
        gen_loss.append(losses[i][0])
        disc_loss.append(losses[i][1])

    plt.figure(figsize=(12, 8))
    plt.plot(gen_loss, 'b-', label='Generator Loss')
    plt.plot(disc_loss, 'r-', label='Discriminator Loss')
    plt.ylabel("Loss value")
    plt.legend(loc='upper right')

    plt.savefig(f"{save_plot_dir}loss_function_{loss}.png")


# First remove directory, then create it with the passed path
def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def add_value_to_avg(old, new_val, n):
    return old + ((new_val - old)/n)