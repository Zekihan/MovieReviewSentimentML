import numpy as np
import random

# HYPERPARAMETERS
from read_data import get_file_data, generate_dictinoary_data, generate_training_data, generate_training_label, \
    shuffle_arrays

input_size = 200
output_size = 1
# embedding_size =
# hidden_layer_size =
# learning_rate =
number_of_epochs = 10
path = "./data"  # please use relative path like this


def activation_function(layer):
    print('a')


def derivation_of_activation_function(signal):
    print('a')


def loss_function(true_labels, probabilities):
    print('a')


def sigmoid(layer):
    print('a')


def derivation_of_loss_function(true_labels, probabilities):
    print('a')


# the derivation should be with respect to the output neurons

def forward_pass(data):
    print('a')

# [hidden_layers] is not an argument, but it is up to you how many hidden layers to implement.
# so replace it with your desired hidden layers

def backward_pass(input_layer, output_layer, loss):
    print('a')


def train(train_data, train_labels, valid_data, valid_labels):
    for epoch in range(number_of_epochs):
        index = 0

        # Same thing about [hidden_layers] mentioned above is valid here also
        for data, labels in zip(train_data, train_labels):
            predictions = forward_pass(data)
            loss_signals = derivation_of_loss_function(labels, predictions)
            backward_pass(data, predictions, loss_signals)
            loss = loss_function(labels, predictions)

            if index % 2000 == 0:  # at each 2000th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_data, valid_labels)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(train_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))

            index += 1

    return loss


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    for data, label in zip(test_data, test_labels):  # Turns through all data
        prediction, _, _ = forward_pass(data)
        predictions.append(prediction)
        labels.append(label)
        avg_loss += np.sum(loss_function(label, prediction))

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0

    # for i in range(len(predictions)):
    # 	if ...
    # 		true_pred += 1

    return true_pred / len(predictions)


if __name__ == "__main__":
    # read data from text files

    # read data from text files
    data_text = get_file_data('./data/reviews.txt')
    word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
    data = generate_training_data(data_text, word_to_index)
    labels_text = get_file_data('./data/labels.txt')
    labels = generate_training_label(labels_text)

    shuffle_arrays([data, labels])

    train_x, train_y = data[int(0.8 * len(data)):-1], labels[int(0.8 * len(labels)):-1]
    test_x, test_y = data[0:int(0.8 * len(data))], labels[0:int(0.8 * len(labels))]

    # Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8 * len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8 * len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8 * len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8 * len(train_y))])

    train(train_x, train_y, valid_x, valid_y)
    print("Test Scores:")
    print(test(test_x, test_y))
