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


def forward_prop(target_word_vector):
    # target_word_vector = x , weight_inp_hidden =  weights for input layer to hidden layer
    hidden_layer = np.dot(weights_input_hidden.T, target_word_vector)

    # weight_hidden_output = weights for hidden layer to output layer
    u = np.dot(weights_hidden_output.T, hidden_layer)

    y_predicted = sigmoid(u)

    return y_predicted, hidden_layer, u


def calculate_error(y_pred, context_words):
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}

    for i, value in enumerate(y_pred):

        if index_of_1_in_context_words.get(i) != None:
            total_error[i] = (value - 1) + ((number_of_1_in_context_vector - 1) * value)
        else:
            total_error[i] = (number_of_1_in_context_vector * value)

    return np.array(total_error)


def calculate_loss(u, ctx):
    total_loss = len(np.where(ctx == 1)[0]) * np.log(np.sum(np.exp(u)))
    return total_loss


def train(word_embedding_dimension, window_size, epochs, training_data, training_labels, learning_rate, valid_data,
          valid_labels, disp='no', interval=-1):
    # For analysis purposes
    global weights_input_hidden, weights_hidden_output
    epoch_loss = []
    weights_1 = []
    weights_2 = []

    for epoch in range(epochs):
        index = 0
        for target, context in zip(training_data, training_labels):
            y_pred, hidden_layer, u = forward_prop(target)

            total_error = calculate_error(y_pred, context)

            weights_input_hidden, weights_hidden_output = backward_prop(
                total_error, hidden_layer, target, learning_rate
            )
            if index % 200 == 0:  # at each 2000th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_data, valid_labels)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(training_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))
            index += 1
        weights_1.append(weights_input_hidden)
        weights_2.append(weights_hidden_output)

    return np.array(weights_1), np.array(weights_2)


def backward_prop(total_error, hidden_layer, target_word_vector,
                  learning_rate):
    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(weights_hidden_output, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)

    # Update weights
    weight_inp_hidden = weights_input_hidden - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = weights_hidden_output - (learning_rate * dl_weight_hidden_output)

    return weight_inp_hidden, weight_hidden_output


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
        text.append(review[:200])
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
        vector = get_one_hot_vectors(line, n_inputs, word_to_index)
        vector_data.append(vector)
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


def sigmoid(layer):
    return 1 / (1 + np.exp(-layer))


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    for data, label in zip(test_data, test_labels):  # Turns through all data
        prediction, _, _ = forward_prop(data)
        predictions.append(sigmoid(prediction))
        labels.append(label)
        avg_loss += np.sum(calculate_loss(label, prediction))

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0

    for i in range(len(predictions)):
        true_pred += 1

    return true_pred / len(predictions)


dimension = 2
epochs = 10
learning_rate = 0.01
window_size = 1

if __name__ == "__main__":
    print('reading reviews')
    data_text = get_file_data('./data/reviews.txt')
    word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
    n_inputs = vocab_size
    data = generate_training_data(data_text, word_to_index)

    weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, dimension))

    weights_hidden_output = np.random.uniform(-1, 1, (dimension, 1))

    print('reading labels')
    labels_text = get_file_data('./data/labels.txt')
    labels = generate_training_label(labels_text)

    print('preparing data')
    shuffle_arrays([data, labels])

    train_x, train_y = data[int(0.8 * len(data)):-1], labels[int(0.8 * len(labels)):-1]
    test_x, test_y = data[0:int(0.8 * len(data))], labels[0:int(0.8 * len(labels))]

    # Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8 * len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8 * len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8 * len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8 * len(train_y))])

    print('training')
    epoch_loss, weights_1, weights_2 = train(dimension, window_size, epochs, train_x, train_y, learning_rate, valid_x,
                                             valid_y,
                                             disp='yes', interval=1)
