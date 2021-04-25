from math import exp
import random
import re
import numpy as np
from gensim.models import word2vec

import torch
from torchtext import data as tdata

SEED = 2019
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
TEXT = tdata.Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = tdata.LabelField(dtype=torch.float, batch_first=True)
fields = [(None, None), ('text', TEXT), ('label', LABEL)]
# loading custom dataset
training_data = tdata.TabularDataset(path='quora.csv', format='csv', fields=fields, skip_header=True)

# print preprocessed text
print(vars(training_data.examples[0]))

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

# HYPERPARAMETERS
input_size = 200
output_size = 1
embedding_size = 200
hidden_layer_size = 2
learning_rate = 0.5
number_of_epochs = 20
path = "./data"

n_output_class = 2
n_inputs = 200 * 200
n_outputs = output_size * n_output_class


def activation_function(weights, layer):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * layer[i]
    return sigmoid(activation)


def derivation_of_activation_function(signal):
    return signal * (1.0 - signal)


def loss_function(true_labels, probabilities):
    return sum([(true_labels[i] - probabilities[i]) ** 2 for i in range(len(true_labels))])


def sigmoid(layer):
    try:
        value = 1.0 / (1.0 + exp(-layer))
    except:
        value = 0.0
    return value


def forward_pass(network, data):
    inputs = data
    for layer in network:
        new_inputs = []
        for neuron in layer:
            neuron['output'] = activation_function(neuron['weights'], inputs)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_pass(network, expected):
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
            neuron['delta'] = errors[j] * derivation_of_activation_function(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train(network, dataset, valid_dataset):
    global loss
    for epoch in range(number_of_epochs):
        index = 0
        loss = 0
        for data in dataset:
            predictions = forward_pass(network, data)
            expected = [0 for i in range(n_outputs)]
            expected[int(data[-1])] = 1
            backward_pass(network, expected)
            loss += loss_function(expected, predictions)
            update_weights(network, data, learning_rate)

            if index % 200 == 0:
                accuracy, loss = test(network, valid_dataset)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(dataset))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))

            index += 1

    return loss


def test(network, test_dataset):
    global data
    avg_loss = 0
    predictions = []
    labels = []
    for data in test_dataset:  # Turns through all data
        prediction = forward_pass(network, data)
        predictions.append(prediction)

        expected = [0 for i in range(n_outputs)]
        expected[int(data[-1])] = 1
        labels.append(expected)

        avg_loss += np.sum(loss_function(expected, prediction))

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(data)


def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
        if np.argmax(true_labels) == np.argmax(predictions):
            true_pred += 1

    return true_pred / len(predictions)


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
        text.append(review.split(' '))
    return text


def word2vec_init():
    data_text = get_file_data('./data/reviews.txt')
    data_text.append(['<padding>', '<unknown>'])
    model = word2vec.Word2Vec(data_text, vector_size=200, window=5, min_count=1, workers=4)
    model.train(data_text, total_examples=len(data_text), epochs=10)
    return model


def generate_training_label(labels_text):
    vector_data = []
    for label in labels_text:
        if label[0] == 'positive':
            vector_data.append([1])
        else:
            vector_data.append([0])
    return vector_data


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def vectorisation_data(model, data_text):
    data_text = np.asarray(data_text)
    new_data = []
    for text in data_text:
        new_text = []
        for word in text[:200]:
            try:
                new_text.append(model.wv[word])
            except:
                new_text.append(model.wv['<unknown>'])
        for _ in range(len(new_text), 200):
            new_text.append(model.wv['<padding>'])
        new_data.append(np.asarray(new_text).flatten())
    return new_data


def part_a():
    model = word2vec_init()

    data_text = get_file_data('./data/reviews.txt')
    labels_text = get_file_data('./data/labels.txt')
    labels = generate_training_label(labels_text)

    train_x, train_y = np.asarray(data_text[0:int(0.1 * len(data_text))]), np.asarray(labels[0:int(0.1 * len(labels))])
    test_x, test_y = vectorisation_data(model, np.asarray(data_text[int(0.8 * len(data_text)):-1])), np.asarray(
        labels[int(0.8 * len(labels)):-1])

    # Training and validation split. (%80-%20)
    valid_x = vectorisation_data(model, np.asarray(train_x[int(0.8 * len(train_x)):-1]))
    valid_y = np.asarray(train_y[int(0.8 * len(train_y)):-1])
    train_x = vectorisation_data(model, np.asarray(train_x[0:int(0.8 * len(train_x))]))
    train_y = np.asarray(train_y[0:int(0.8 * len(train_y))])

    train_dataset = np.concatenate((train_x, train_y), axis=1)
    valid_dataset = np.concatenate((valid_x, valid_y), axis=1)
    test_dataset = np.concatenate((test_x, test_y), axis=1)

    n_inputs = len(train_dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in train_dataset]))
    network = initialize_network(n_inputs, hidden_layer_size, n_outputs)

    train(network, train_dataset, valid_dataset)
    print("Test Scores:")
    print(test(network, test_dataset))


def part_b():
    data_text = get_file_data('./data/reviews.txt')
    labels_text = get_file_data('./data/labels.txt')


if __name__ == "__main__":
    # part_a()
    part_b()
