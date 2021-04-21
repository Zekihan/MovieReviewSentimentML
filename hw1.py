import re

import numpy as np
import random

# HYPERPARAMETERS
input_size = 200
output_size = 1
embedding_size = 200
hidden_layer_size = 16
learning_rate = 1
number_of_epochs = 10
path = "./data"  # please use relative path like this

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def activation_function(layer):
    print("activation_function")


def derivation_of_activation_function(signal):
    print("derivation_of_activation_function")


def loss_function(true_labels, probabilities):
    print("loss_function")


def sigmoid(layer):
    print("sigmoid")


def derivation_of_loss_function(true_labels, probabilities):
    print("derivation_of_loss_function")


# the derivation should be with respect to the output neurons

def forward_pass(data):
    print("forward_pass")


# [hidden_layers] is not an argument, but it is up to you how many hidden layers to implement.
# so replace it with your desired hidden layers

def backward_pass(input_layer, hidden_layers, output_layer, loss):
    print("backward_pass")


def train(train_data, train_labels, valid_data, valid_labels):
    for epoch in range(number_of_epochs):
        index = 0

        # Same thing about [hidden_layers] mentioned above is valid here also
        for data, labels in zip(train_data, train_labels):
            predictions, [hidden_layers] = forward_pass(data)
            loss_signals = derivation_of_loss_function(labels, predictions)
            backward_pass(data, [hidden_layers], predictions, loss_signals)
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

    for i in range(len(predictions)):
        true_pred += 1

    return true_pred / len(predictions)


def get_file_data(path):
    with open(path) as f:
        file_contents = f.read()
    text = []
    for val in file_contents.split('\n'):
        sent = re.findall("[A-Za-z]+", val)
        line = ''
        for words in sent:
            if len(words) > 1:
                line = line + words + ' '
        if len(line) > 1:
            line = line[:-1]
            text.append(line)
    return text


def generate_dictinoary_data(text):
    word_to_index = dict()
    index_to_word = dict()
    corpus = []
    count = 0
    vocab_size = 0

    for row in text:
        for word in row.split():
            word = word.lower()
            corpus.append(word)
            if word_to_index.get(word) == None:
                word_to_index.update({word: count})
                index_to_word.update({count: word})
                count += 1
    vocab_size = len(word_to_index)
    length_of_corpus = len(corpus)

    return word_to_index, index_to_word, corpus, vocab_size, length_of_corpus


def get_one_hot_vectors(context_words, vocab_size, word_to_index):
    ctxt_word_vector = np.zeros(vocab_size)
    words = context_words.split(' ')[0:200]
    for word in words:
        index_of_word_dictionary = word_to_index.get(word)
        if index_of_word_dictionary is not None:
            ctxt_word_vector[index_of_word_dictionary] = 1

    return ctxt_word_vector


def generate_training_data(text, word_to_index, vocab_size):
    vector_data = []
    for line in text:
        vector = get_one_hot_vectors(line, vocab_size, word_to_index)
        vector_data.append(vector)
    return vector_data


def generate_training_label(labels_text):
    vector_data = []
    for label in labels_text:
        if label == 'positive':
            vector_data.append(1)
        else:
            vector_data.append(0)
    return vector_data


def shuffle_arrays(arrays, set_seed=-1):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)


if __name__ == "__main__":
    # read data from text files
    data_text = get_file_data('./data/reviews.txt')
    word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
    data = generate_training_data(data_text, word_to_index, vocab_size)
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
