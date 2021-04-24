import numpy as np
import random
import re

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
hidden_layer_size = 16
learning_rate = 0.5
number_of_epochs = 50
path = "./data"  # please use relative path like this

n_inputs = 200
n_hidden = 2
n_outputs = 1
network = []


def initialize_parameters():
    local_network = list()
    hidden_layer = [{'weights': [random.random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
    local_network.append(hidden_layer)
    output_layer = [{'weights': [random.random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    local_network.append(output_layer)
    return local_network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))


def activation_function(layer):
    return 1 / (1 + np.exp(-layer))


def derivation_of_activation_function(signal):
    s = 1 / (1 + np.exp(-signal))
    return s * (1 - s)


def loss_function(true_labels, probabilities):
    sum_score = 0.0
    for i in range(len(true_labels)):
        sum_score += true_labels[i] * np.log(1e-15 + probabilities[i])
    mean_sum_score = 1.0 / len(true_labels) * sum_score
    return -mean_sum_score


def sigmoid(layer):
    return 1 / (1 + np.exp(-layer))


def derivation_of_loss_function(true_labels, probabilities):
    sum_score = 0.0
    for i in range(len(true_labels)):
        sum_score += true_labels[i] * np.log(1e-15 + probabilities[i])
    mean_sum_score = 1.0 / len(true_labels) * sum_score
    return -mean_sum_score


# the derivation should be with respect to the output neurons

def forward_pass(inputs):
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def transfer_derivative(output):
    return output * (1.0 - output)

def backward_pass(data, predictions, loss_signals):
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

    for i in range(len(predictions)):
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
            vector_data.append([1])
        else:
            vector_data.append([0])
    return vector_data


def shuffle_arrays(arrays, set_seed=-1):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        random_state = np.random.RandomState(seed)
        random_state.shuffle(arr)


if __name__ == "__main__":
    # read data from text files
    data_text = get_file_data('./data/reviews.txt')
    word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
    n_inputs = vocab_size
    data = generate_training_data(data_text, word_to_index)
    labels_text = get_file_data('./data/labels.txt')
    labels = generate_training_label(labels_text)

    shuffle_arrays([data, labels])

    network = initialize_parameters()

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
