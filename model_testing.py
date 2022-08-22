import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_scores(histories, model_index):

    acc, loss, val_acc, val_loss = [], [], [], []

    for i in range(len(histories)):
        acc.append(np.ravel(histories[i].history['accuracy']))
        loss.append(np.ravel(histories[i].history['loss']))
        val_acc.append(np.ravel(histories[i].history['val_accuracy']))
        val_loss.append(np.ravel(histories[i].history['val_loss']))

    acc = np.ravel(acc)
    val_acc = np.ravel(val_acc)
    loss = np.ravel(loss)
    val_loss = np.ravel(val_loss)

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
