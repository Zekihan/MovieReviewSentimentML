import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def get_file_data(path, stop_word_removal='no'):
    with open(path) as f:
        lines = f.readlines()
    text = []
    lines = lines[:500]
    for line in lines:
        review = line[:-1]
        review = review[:200]
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


text = ['Best way to success is through hardwork and persistence']
word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(text)
print('Number of unique words:', vocab_size)
print('word_to_index : ', word_to_index)
print('index_to_word : ', index_to_word)
print('corpus:', corpus)
print('Length of corpus :', length_of_corpus)


def get_one_hot_vectors(target_word, context_words, vocab_size, word_to_index):
    # Create an array of size = vocab_size filled with zeros
    trgt_word_vector = np.zeros(vocab_size)

    # Get the index of the target_word according to the dictionary word_to_index.
    # If target_word = best, the index according to the dictionary word_to_index is 0.
    # So the one hot vector will be [1, 0, 0, 0, 0, 0, 0, 0, 0]
    index_of_word_dictionary = word_to_index.get(target_word)

    # Set the index to 1
    trgt_word_vector[index_of_word_dictionary] = 1

    # Repeat same steps for context_words but in a loop
    ctxt_word_vector = np.zeros(vocab_size)

    for word in context_words:
        index_of_word_dictionary = word_to_index.get(word)
        ctxt_word_vector[index_of_word_dictionary] = 1

    return trgt_word_vector, ctxt_word_vector


# Note : Below comments for trgt_word_index, ctxt_word_index are with the above sample text for understanding the code flow

def generate_training_data(corpus, window_size, vocab_size, word_to_index, length_of_corpus, sample=None):
    training_data = []
    training_sample_words = []
    for i, word in enumerate(corpus):

        index_target_word = i
        target_word = word
        context_words = []

        # when target word is the first word
        if i == 0:

            # trgt_word_index:(0), ctxt_word_index:(1,2)
            context_words = [corpus[x] for x in range(i + 1, window_size + 1)]

            # when target word is the last word
        elif i == len(corpus) - 1:

            # trgt_word_index:(9), ctxt_word_index:(8,7), length_of_corpus = 10
            context_words = [corpus[x] for x in range(length_of_corpus - 2, length_of_corpus - 2 - window_size, -1)]

        # When target word is the middle word
        else:

            # Before the middle target word
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size, -1):
                if x >= 0:
                    context_words.extend([corpus[x]])

            # After the middle target word
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < len(corpus):
                    context_words.extend([corpus[x]])

        trgt_word_vector, ctxt_word_vector = get_one_hot_vectors(target_word, context_words, vocab_size, word_to_index)
        training_data.append([trgt_word_vector, ctxt_word_vector])

        if sample is not None:
            training_sample_words.append([target_word, context_words])

    return training_data, training_sample_words


text = ['Best way to success is through hardwork and persistence']
word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(text)

window_size = 2
training_data, training_sample_words = generate_training_data(corpus, 2, vocab_size, word_to_index, length_of_corpus,
                                                              'yes')

for i in range(len(training_data)):
    print('*' * 50)
    print('Target word:%s . Target vector: %s ' % (training_sample_words[i][0], training_data[i][0]))
    print('Context word:%s . Context  vector: %s ' % (training_sample_words[i][1], training_data[i][1]))


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
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}

    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update({index: 'yes'})

    number_of_1_in_context_vector = len(index_of_1_in_context_words)

    for i, value in enumerate(y_pred):

        if index_of_1_in_context_words.get(i) != None:
            total_error[i] = (value - 1) + ((number_of_1_in_context_vector - 1) * value)
        else:
            total_error[i] = (number_of_1_in_context_vector * value)

    return np.array(total_error)


def calculate_loss(u, ctx):
    sum_1 = 0
    for index in np.where(ctx == 1)[0]:
        sum_1 = sum_1 + u[index]

    sum_1 = -sum_1
    sum_2 = len(np.where(ctx == 1)[0]) * np.log(np.sum(np.exp(u)))

    total_loss = sum_1 + sum_2
    return total_loss


def train(word_embedding_dimension, window_size, epochs, training_data, learning_rate, disp='no', interval=-1):
    weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, word_embedding_dimension))
    weights_hidden_output = np.random.uniform(-1, 1, (word_embedding_dimension, vocab_size))

    # For analysis purposes
    epoch_loss = []
    weights_1 = []
    weights_2 = []

    for epoch in range(epochs):
        loss = 0

        for target, context in training_data:
            y_pred, hidden_layer, u = forward_prop(weights_input_hidden, weights_hidden_output, target)

            total_error = calculate_error(y_pred, context)

            weights_input_hidden, weights_hidden_output = backward_prop(
                weights_input_hidden, weights_hidden_output, total_error, hidden_layer, target, learning_rate
            )

            loss_temp = calculate_loss(u, context)
            loss += loss_temp

        epoch_loss.append(loss)
        weights_1.append(weights_input_hidden)
        weights_2.append(weights_hidden_output)

        if disp == 'yes':
            if epoch == 0 or epoch % interval == 0 or epoch == epochs - 1:
                print('Epoch: %s. Loss:%s' % (epoch, loss))
    return epoch_loss, np.array(weights_1), np.array(weights_2)


# Input vector, returns nearest word(s)
def cosine_similarity(word, weight, word_to_index, vocab_size, index_to_word):
    # Get the index of the word from the dictionary
    index = word_to_index[word]

    # Get the correspondin weights for the word
    word_vector_1 = weight[index]

    word_similarity = {}

    for i in range(vocab_size):
        word_vector_2 = weight[i]

        theta_sum = np.dot(word_vector_1, word_vector_2)
        theta_den = np.linalg.norm(word_vector_1) * np.linalg.norm(word_vector_2)
        theta = theta_sum / theta_den

        word = index_to_word[i]
        word_similarity[word] = theta

    return word_similarity  # words_sorted


def print_similar_words(top_n_words, weight, msg, words_subset):
    columns = []

    for i in range(0, len(words_subset)):
        columns.append('similar:' + str(i + 1))

    df = pd.DataFrame(columns=columns, index=words_subset)
    df.head()

    row = 0
    for word in words_subset:

        # Get the similarity matrix for the word: word
        similarity_matrix = cosine_similarity(word, weight, word_to_index, vocab_size, index_to_word)
        col = 0

        # Sort the top_n_words
        words_sorted = dict(sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)[1:top_n_words + 1])

        # Create a dataframe to display the similarity matrix
        for similar_word, similarity_value in words_sorted.items():
            df.iloc[row][col] = (similar_word, round(similarity_value, 2))
            col += 1
        row += 1
    styles = [dict(selector='caption',
                   props=[('text-align', 'center'), ('font-size', '20px'), ('color', 'red')])]
    df = df.style.set_properties(**
                                 {'color': 'green', 'border-color': 'blue', 'font-size': '14px'}
                                 ).set_table_styles(styles).set_caption(msg)
    return df


def plot_epoch_loss(lbl, loss_epoch, plot_title):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    i = 0
    fig = plt.figure(figsize=(10, 5), facecolor='w', edgecolor='k', dpi=80)
    plt.suptitle('Epoch vs Loss', fontsize=16)

    for key, loss in loss_epoch.items():
        epoch_count = range(1, len(loss) + 1)

        plt.plot(epoch_count, loss, 'r-', color=colors[i], linewidth=2.0, label=lbl + str(key))

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        i += 1

    plt.legend(framealpha=1, frameon=True, fontsize='large', edgecolor="inherit", shadow=True)
    plt.title(plot_title)
    plt.show()
    plt.close()

window_size = 2
dimension = 70
epochs = 1000
learning_rate = 0.01
text = ['Best way to success is through hardwork and persistence']
data_text = get_file_data('./data/reviews.txt')

print('gen_dict')
word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
print('gen_train')
training_data, training_sample_words = generate_training_data(corpus, window_size, vocab_size, word_to_index,
                                                              length_of_corpus)

print('train')
epoch_loss, weights_1, weights_2 = train(dimension, window_size, epochs, training_data, learning_rate, disp='yes')
print(epoch_loss)
