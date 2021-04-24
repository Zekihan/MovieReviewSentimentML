from math import exp
from random import seed
from random import random
import numpy as np
import re
import time

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

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
REPLACE_STOP_WORDS = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')

dimension = 2
epochs = 10
learning_rate = 0.01
window_size = 1


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train_x, train_y, valid_data, valid_labels, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        index = 0
        for row, label in zip(train_x, train_y):
            outputs = forward_propagate(network, row)
            backward_propagate_error(network, label)
            update_weights(network, row, l_rate)
            if index % 100 == 0 and index != 0:
                accuracy, loss = test(network, valid_data, valid_labels)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(train_x))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))
            index += 1


def get_file_data(path):
    with open(path) as f:
        lines = f.readlines()
    text = []
    for line in lines:
        review = line[:-1]
        review = REPLACE_NO_SPACE.sub("", review.lower())
        review = REPLACE_NO_SPACE.sub(" ", review.lower())
        review = REPLACE_STOP_WORDS.sub(" ", review.lower())
        review = ' '.join([w for w in review.split() if len(w) > 1])
        text.append(review)
    return text


def generate_dictinoary_data(text):
    word_to_index = dict()
    index_to_word = dict()
    corpus = []
    count = 0
    for row in text:
        for word in row.split():
            word = word.lower()
            corpus.append(word)
            if word_to_index.get(word) is None:
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


def generate_training_data(text, word_to_index):
    vector_data = []
    for line in text:
        vector = []
        for word in line.split(" ")[:200]:
            vector.append(word_to_index[word])
        vector_data.append(vector)
        for i in range(len(vector), 200):
            vector.append(-1)

    return vector_data


def generate_training_label(labels_text):
    vector_data = []
    for label in labels_text:
        if label == 'positive':
            vector_data.append([1, 0])
        else:
            vector_data.append([0, 1])
    return vector_data


def shuffle_arrays(arrays, set_seed=-1):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        random_state = np.random.RandomState(seed)
        random_state.shuffle(arr)


def sigmoid(layer):
    return 1 / (1 + np.exp(-layer))


def test(network, test_data, test_labels):
    avg_loss = 0
    predictions = []

    for data, label in zip(test_data, test_labels):  # Turns through all data
        prediction = forward_propagate(network, data)
        predictions.append(prediction)

    accuracy_score = accuracy(test_labels, predictions)
    return accuracy_score, 0


def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
        if np.argmax(true_labels[i]) == np.argmax(predictions[i]):
            true_pred += 1

    return true_pred / len(predictions)


# Test training backprop algorithm
seed(1)
print('reading reviews')
data_text = get_file_data('./data/reviews.txt')
word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
n_inputs = 200
n_outputs = 2  # len(train_y[0])
network = initialize_network(n_inputs, 128, n_outputs)
data = generate_training_data(data_text, word_to_index)

weights_input_hidden = np.random.uniform(-1, 1, (200, dimension))

weights_hidden_output = np.random.uniform(-1, 1, (dimension, 1))

print('reading labels')
labels_text = get_file_data('./data/labels.txt')
labels = generate_training_label(labels_text)

print('preparing data')
shuffle_arrays([data, labels])

train_x, train_y = data[0:int(0.1 * len(data))], labels[0:int(0.1 * len(labels))]
test_x, test_y = data[int(0.05 * len(data)):-1], labels[int(0.05 * len(labels)):-1]

# Training and validation split. (%80-%20)
valid_x = np.asarray(train_x[int(0.8 * len(train_x)):-1])
valid_y = np.asarray(train_y[int(0.8 * len(train_y)):-1])
train_x = np.asarray(train_x[0:int(0.8 * len(train_x))])
train_y = np.asarray(train_y[0:int(0.8 * len(train_y))])

print('training')
train_network(network, train_x, train_y, valid_x, valid_y, 0.01, 20, n_outputs)
for layer in network:
    print(layer)
