import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

N_OF_FOLDS = 2
SAVE_PLOT_DIR = "plots/"


def plot_scores(histories, model_index):

    create_dir(SAVE_PLOT_DIR)
    acc, loss, val_acc, val_loss = [], [], [], []

    # For each epoch there are 5 values of training loss, validation loss,
    # training accuracy and validation accuracy
    for i in range(len(histories)):
        epoch = histories[i]

        for i in range(N_OF_FOLDS):
            loss.append(epoch[0][i][0])
            acc.append(epoch[0][i][1])
            val_loss.append(epoch[1][i][0])
            val_acc.append(epoch[1][i][1])

        f, axis = plt.subplots(2, 1)
        x = list(range(N_OF_FOLDS))
        axis[0].set_title('Training and validation loss')
        axis[0].plot(x, loss, color='blue', label='train')
        axis[0].plot(x, val_loss, color='orange', label='test')

        axis[1].set_title('Training and validation accuracy')
        axis[1].plot(x, acc, color='blue', label='train')
        axis[1].plot(x, val_acc, color='orange', label='test')

        plt.savefig(f"{SAVE_PLOT_DIR}model{model_index}_epoch{i}_plot.png")

    print("Plots correctly saved.")

# First remove directory, then create it with the passed path
def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)