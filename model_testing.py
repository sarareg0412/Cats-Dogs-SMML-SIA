import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

N_OF_FOLDS = 5
EPOCHS = 2
SAVE_PLOT_DIR = "plots/"


def plot_scores(histories, model_index):

    create_dir(SAVE_PLOT_DIR)
    acc, loss, val_acc, val_loss = [], [], [], []

    # For each fold there are N_EPOCH values of training loss, validation loss,
    # training accuracy and validation accuracy
    for i in range(N_OF_FOLDS):

        loss = histories[i]['loss']
        acc = histories[i]['accuracy']
        val_loss = histories[i]['val_loss']
        val_acc = histories[i]['val_accuracy']

        plt.figure(figsize=(12, 8))
        x = range(EPOCHS)
        f, axis = plt.subplots(2, 1)
        axis[0].set_title('Training and validation loss')
        axis[0].plot(x, loss, color='blue', label='Training Loss')
        axis[0].plot(x, val_loss, color='orange', label='Validation Loss')
        axis[0].legend(loc='lower right')

        axis[1].set_title('Training and validation accuracy')
        axis[1].plot(x, acc, color='blue', label='Training Accuracy')
        axis[1].plot(x, val_acc, color='orange', label='Validation Accuracy')
        axis[0].legend(loc='lower right')

        plt.savefig(f"{SAVE_PLOT_DIR}model{model_index}_fold{i}_total_plot.png")

    accs = [lambda: sum(histories[i]['val_accuracy']) / EPOCHS for i in range(N_OF_FOLDS)]

    plt.figure(figsize=(12, 8))
    plt.plot(range(N_OF_FOLDS), accs, label='Validation Accuracy for each fold')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.savefig(f"{SAVE_PLOT_DIR}model{model_index}_K_fold_accuracy_plot.png")
    print("Plots correctly saved.")

def plot_array(array, title, name):
    plt.title(title)
    plt.plot(list(range(len(array))), array, color='blue', label='train')

    plt.savefig(f"{name}_plot.png")

# First remove directory, then create it with the passed path
def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)