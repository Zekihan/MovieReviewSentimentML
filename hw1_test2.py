import re

import numpy as np

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


def get_data_set(data_text):
    from gensim.models import word2vec

    model = word2vec.Word2Vec.load("word2vec.model")

    data_text = np.asarray(data_text)
    new_data = []
    for text in data_text:
        new_text = []
        for word in text[:200]:
            new_text.append(model.wv[word])
        for _ in range(len(new_text), 200):
            new_text.append(model.wv['<padding>'])
        new_data.append(np.asarray(new_text).flatten())
    return new_data


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


def generate_training_label(labels_text):
    vector_data = []
    for label in labels_text:
        if label[0] == 'positive':
            vector_data.append([1])
        else:
            vector_data.append([0])
    return vector_data


def sigmoid(x, Derivative=False):
    if not Derivative:
        return 1 / (1 + np.exp(-x))
    else:
        out = sigmoid(x)
        return out * (1 - out)


class backPropNN:
    """Class defining a NN using Back Propagation"""

    # Class Members (internal variables that are accessed with backPropNN.member)
    numLayers = 0
    shape = None
    weights = []

    # Class Methods (internal functions that can be called)

    def __init__(self, numNodes):
        """Initialise the NN - setup the layers and initial weights"""

        # Layer information
        self.numLayers = len(numNodes) - 1
        self.shape = numNodes

        for (l1, l2) in zip(numNodes[:-1], numNodes[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(l2, l1 + 1)))
        self._layerInput = []
        self._layerOutput = []

    # Forward Pass method
    def FP(self, input):
        numExamples = input.shape[0]

        # Clean away the values from the previous layer
        self._layerInput = []
        self._layerOutput = []

        layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, numExamples])]))
        for index in range(self.numLayers):
            # Get input to the layer
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, numExamples])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, numExamples])]))
        self._layerInput.append(layerInput)
        self._layerOutput.append(sigmoid(layerInput))
        return self._layerOutput[-1].T

    # TrainEpoch method
    def backProp(self, input, target, trainingRate=0.2):
        """Get the error, deltas and back propagate to update the weights"""
        delta = []
        numExamples = input.shape[0]

        # Do the forward pass
        self.FP(input)

        for index in reversed(range(self.numLayers)):
            if index == self.numLayers - 1:
                # If the output layer, then compare to the target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * sigmoid(self._layerInput[index], True))
            else:
                # If a hidden layer. compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * sigmoid(self._layerInput[index], True))


from math import exp
from random import seed
from random import random


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
    try:
        value = 1.0 / (1.0 + exp(-activation))
    except:
        value = 0.0
    return value


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation =
            neuron['output'] = transfer(activate(neuron['weights'], inputs))
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
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        index = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            index += 1
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


if __name__ == '__main__':
    data_text = get_file_data('./data/reviews.txt')
    data = get_data_set(data_text)
    labels_text = get_file_data('./data/labels.txt')
    labels = generate_training_label(labels_text)

    train_x, train_y = np.asarray(data[0:int(0.1 * len(data))]), np.asarray(labels[0:int(0.1 * len(labels))])
    test_x, test_y = np.asarray(data[int(0.8 * len(data)):-1]), np.asarray(labels[int(0.8 * len(labels)):-1])

    # Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8 * len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8 * len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8 * len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8 * len(train_y))])

    X = train_x
    Y = train_y.astype(int)
    # Test training backprop algorithm
    seed(1)
    dataset = np.concatenate((X, Y), axis=1)
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, 0.5, 20, n_outputs)
    for layer in network:
        print(layer)
