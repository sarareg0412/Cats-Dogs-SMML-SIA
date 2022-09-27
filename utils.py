import os
import matplotlib.pyplot as plt

N_OF_FOLDS = 5


def plot_scores(histories, model_index, save_plot_dir, epochs, batch_size):

    train_acc = []
    val_acc = []

    # accs contains the average of the validation accuracy for each fold
    for i in range(N_OF_FOLDS):
        train_acc.append(sum(histories[i]['accuracy']) / len(histories[i]['accuracy']))
        val_acc.append(sum(histories[i]['val_accuracy']) / len(histories[i]['val_accuracy']))

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, N_OF_FOLDS + 1), train_acc, 'bo--', label="Training Accuracy")
    plt.plot(range(1, N_OF_FOLDS + 1), val_acc, 'ro--', label="Validation Accuracy")
    plt.legend(loc='lower right')
    plt.title('Average accuracy')
    plt.ylabel("Accuracy value")
    plt.xlabel("Fold")
    plt.text(3, min(val_acc), 'AVG VAL_ACCURACY: {:.4f}'.format(sum(val_acc) / len(val_acc)))

    for x, y in zip(range(1, N_OF_FOLDS + 1), val_acc):
        label = "{:.4f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    for x, y in zip(range(1, N_OF_FOLDS + 1), train_acc):
        label = "{:.4f}".format(y)

        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.savefig(f"{save_plot_dir}model{model_index}_{epochs}E_{batch_size}B_accuracy_plot.png")
    print("Plots correctly saved.")


def plot_losses(losses, save_plot_dir, loss, epochs, batch):
    gen_loss, r_disc_loss, f_disc_loss = [], [], []
    for i in range(len(losses)):
        gen_loss.append(losses[i][0])
        r_disc_loss.append(losses[i][1])
        f_disc_loss.append(losses[i][2])

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, epochs + 1), gen_loss, 'b.--', label='Generator Loss')
    plt.plot(range(1, epochs + 1), r_disc_loss, 'r.--', label='Discriminator Loss on Real Images')
    plt.plot(range(1, epochs + 1), f_disc_loss, 'g.--', label='Discriminator Loss on Fake Images')
    plt.ylabel("Loss value")
    plt.xlabel("Epoch")
    plt.legend(loc='center right')
    name = f"{save_plot_dir+loss}/{loss}_{epochs}E_{batch}B.png"
    remove_if_exists(name)
    plt.savefig(name)


# Create directory if it doesn't exist, with the passed path
def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def add_value_to_avg(old, new_val, n):
    return old + ((new_val - old)/n)


def remove_if_exists(name):
    # If the picture already exists, first remove it
    if os.path.isfile(name):
        os.remove(name)