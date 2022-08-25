import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_scores(histories, model_index):

    acc, loss, val_acc, val_loss = [], [], [], []

    for i in range(len(histories)):
        loss.append(histories[0][i][0])
        acc.append(histories[0][i][1])
        val_loss.append(histories[1][i][0])
        val_acc.append(histories[1][i][1])

    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(loss, color='blue', label='train')
    plt.plot(val_loss, color='orange', label='test')

    plt.subplot(211)
    plt.title('Classification Accuracy')
    plt.plot(acc, color='green', label='train')
    plt.plot(val_acc, color='yellow', label='test')

    #plt.legend("Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy")

    plt.savefig("model" + str(model_index) + '_plot.png')
    plt.show()
