import numpy as np
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

np.random.seed(42)

window_size = 2
n_embedding = 10
n_iter = 50
learning_rate = 0.05


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


def get_one_hot_vectors(target_word, vocab_size, word_to_index):
    ctxt_word_vector = np.zeros(vocab_size)
    for word in target_word:
        index_of_word_dictionary = word_to_index.get(word)
        ctxt_word_vector[index_of_word_dictionary] = 1

    return ctxt_word_vector


def generate_training_data(word_to_index, index_to_word, vocab_size, data_text, labels):
    training_data = []
    for i, text in enumerate(data_text):
        context_words = []
        trgt_word_vector = get_one_hot_vectors(data_text[i], vocab_size, word_to_index)
        training_data.append([trgt_word_vector, labels[i]])
    return training_data


def forward_prop(weight_inp_hidden, weight_hidden_output, target_word_vector):
    # target_word_vector = x , weight_inp_hidden =  weights for input layer to hidden layer
    hidden_layer = np.dot(weight_inp_hidden.T, target_word_vector)

    # weight_hidden_output = weights for hidden layer to output layer
    u = np.dot(weight_hidden_output.T, hidden_layer)

    y_predicted = softmax(u)

    return y_predicted, hidden_layer, u


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def backward_prop(weight_inp_hidden, weight_hidden_output, total_error, hidden_layer, target_word_vector,
                  learning_rate):
    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(weight_hidden_output, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)

    # Update weights
    weight_inp_hidden = weight_inp_hidden - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = weight_hidden_output - (learning_rate * dl_weight_hidden_output)

    return weight_inp_hidden, weight_hidden_output


def calculate_error(y_pred, context_words):
    return np.sqrt((y_pred[0] - context_words[0]) ** 2)


def calculate_loss(u, ctx):
    sum_1 = 0
    for index in np.where(ctx == 1)[0]:
        sum_1 = sum_1 + u[index]

    sum_1 = -sum_1
    sum_2 = len(np.where(ctx == 1)[0]) * np.log(np.sum(np.exp(u)))

    total_loss = sum_1 + sum_2
    return total_loss


def train(word_embedding_dimension, window_size, epochs, training_data, learning_rate, valid_d,
          disp='no',
          interval=-1):
    weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, word_embedding_dimension))
    weights_hidden_output = np.random.uniform(-1, 1, (word_embedding_dimension, 1))

    # For analysis purposes
    epoch_loss = []
    weights_1 = []
    weights_2 = []

    for epoch in range(epochs):
        loss = 0
        index = 0
        for target, context in training_data:
            y_pred, hidden_layer, u = forward_prop(weights_input_hidden, weights_hidden_output, target)

            total_error = calculate_error(y_pred, context)

            weights_input_hidden, weights_hidden_output = backward_prop(
                weights_input_hidden, weights_hidden_output, total_error, hidden_layer, target, learning_rate
            )

            loss_temp = calculate_loss(u, context)
            loss += loss_temp

            if index % 2000 == 0:  # at each 2000th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_d, weights_input_hidden, weights_hidden_output)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(training_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))
            index += 1
        epoch_loss.append(loss)
        weights_1.append(weights_input_hidden)
        weights_2.append(weights_hidden_output)

        if disp == 'yes':
            if epoch == 0 or epoch % interval == 0 or epoch == epochs - 1:
                print('Epoch: %s. Loss:%s' % (epoch, loss))
    return epoch_loss, np.array(weights_1), np.array(weights_2)


def test(valid_d, weights_input_hidden, weights_hidden_output):
    avg_loss = 0
    predictions = []
    labels = []

    for data, label in valid_d:  # Turns through all data
        y_pred, hidden_layer, u = forward_prop(weights_input_hidden, weights_hidden_output, data)
        predictions.append(y_pred)
        labels.append(label)
        avg_loss += np.sum(calculate_loss(label, y_pred))

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(valid_d)


def accuracy(true_labels, predictions):
    true_pred = 0

    # for i in range(len(predictions)):
    	# if ...
    	# 	true_pred += 1

    return true_pred / len(predictions)


window_size = 2
epochs = 100
learning_rate = 0.01
dimension = 20
data_text = get_file_data('./data/reviews.txt')

labels_text = get_file_data('./data/labels.txt')
labels = generate_training_label(labels_text)

word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
vocab_size=2000
training_data = generate_training_data(word_to_index, index_to_word, vocab_size, data_text,
                                       labels)

train_d = np.asarray(training_data[0:int(0.8 * len(training_data))])
test_d = np.asarray(training_data[int(0.8 * len(training_data)):-1])

# Training and validation split. (%80-%20)
valid_d = np.asarray(train_d[int(0.8 * len(train_d)):-1])
train_d = np.asarray(train_d[0:int(0.8 * len(train_d))])

epoch_loss, weights_1, weights_2 = train(dimension, window_size, epochs, train_d, learning_rate, valid_d,
                                         disp='yes')

a = 5
