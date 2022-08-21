import matplotlib.pyplot as plt


def plot_scores(scores):
    train = scores[0]           # Accuracy
    validation = scores[1]      # Loss
    plt.subplot(1, 1, 1)
    print(train)
    print(validation)
    plt.plot(train, color='blue', label='train')
    plt.plot(validation, color='red', label='validation')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    print("Ok")
