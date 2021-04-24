import numpy as np  # import numpy library

from test2 import get_file_data, generate_dictinoary_data, generate_training_label, \
    shuffle_arrays


def initialize_parameters(n_in, n_out, ini_type='plain'):
    """
    Helper function to initialize some form of random weights and Zero biases
    Args:
        n_in: size of input layer
        n_out: size of output/number of neurons
        ini_type: set initialization type for weights
    Returns:
        params: a dictionary containing W and b
    """

    params = dict()  # initialize empty dictionary of neural net parameters W and b

    if ini_type == 'plain':
        params['W'] = np.random.randn(n_out, n_in) * 0.01  # set weights 'W' to small random gaussian
    elif ini_type == 'xavier':
        params['W'] = np.random.randn(n_out, n_in) / (np.sqrt(n_in))  # set variance of W to 1/n
    elif ini_type == 'he':
        # Good when ReLU used in hidden layers
        # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # Kaiming He et al. (https://arxiv.org/abs/1502.01852)
        # http: // cs231n.github.io / neural - networks - 2 /  # init
        params['W'] = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)  # set variance of W to 2/n

    params['b'] = np.zeros((n_out, 1))  # set bias 'b' to zeros

    return params


class LinearLayer:
    """
        This Class implements all functions to be executed by a linear layer
        in a computational graph
        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
                      Opitons are: plain, xavier and he
        Methods:
            forward(A_prev)
            backward(upstream_grad)
            update_params(learning_rate)
    """

    def __init__(self, input_shape, n_out, ini_type="plain"):
        """
        The constructor of the LinearLayer takes the following parameters
        Args:
            input_shape: input shape of Data/Activations
            n_out: number of neurons in layer
            ini_type: initialization type for weight parameters, default is "plain"
        """

        self.m = input_shape[1]  # number of examples in training data
        # `params` store weights and bias in a python dictionary
        self.params = initialize_parameters(input_shape[0], n_out, ini_type)  # initialize weights and bias
        self.Z = np.zeros((self.params['W'].shape[0], input_shape[1]))  # create space for resultant Z output

    def forward(self, A_prev):
        """
        This function performs the forwards propagation using activations from previous layer
        Args:
            A_prev:  Activations/Input Data coming into the layer from previous layer
        """

        self.A_prev = A_prev  # store the Activations/Training Data coming in
        self.Z = np.dot(self.params['W'], self.A_prev) + self.params['b']  # compute the linear function

    def backward(self, upstream_grad):
        """
        This function performs the back propagation using upstream gradients
        Args:
            upstream_grad: gradient coming in from the upper layer to couple with local gradient
        """

        # derivative of Cost w.r.t W
        self.dW = np.dot(upstream_grad, self.A_prev.T)

        # derivative of Cost w.r.t b, sum across rows
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)

        # derivative of Cost w.r.t A_prev
        self.dA_prev = np.dot(self.params['W'].T, upstream_grad)

    def update_params(self, learning_rate=0.1):
        """
        This function performs the gradient descent update
        Args:
            learning_rate: learning rate hyper-param for gradient descent, default 0.1
        """

        self.params['W'] = self.params['W'] - learning_rate * self.dW  # update weights
        self.params['b'] = self.params['b'] - learning_rate * self.db  # update bias(es)


class SigmoidLayer:
    """
    This file implements activation layers
    inline with a computational graph model
    Args:
        shape: shape of input to the layer
    Methods:
        forward(Z)
        backward(upstream_grad)
    """

    def __init__(self, shape):
        """
        The consturctor of the sigmoid/logistic activation layer takes in the following arguments
        Args:
            shape: shape of input to the layer
        """
        self.A = np.zeros(shape)  # create space for the resultant activations

    def forward(self, Z):
        """
        This function performs the forwards propagation step through the activation function
        Args:
            Z: input from previous (linear) layer
        """
        self.A = 1 / (1 + np.exp(-Z))  # compute activations

    def backward(self, upstream_grad):
        """
        This function performs the  back propagation step through the activation function
        Local gradient => derivative of sigmoid => A*(1-A)
        Args:
            upstream_grad: gradient coming into this layer from the layer above
        """
        # couple upstream gradient with local gradient, the result will be sent back to the Linear layer
        self.dZ = upstream_grad * self.A * (1 - self.A)


def compute_bce_cost(Y, P_hat):
    """
    This function computes Binary Cross-Entropy(bce) Cost and returns the Cost and its
    derivative.
    This function uses the following Binary Cross-Entropy Cost defined as:
    => (1/m) * np.sum(-Y*np.log(P_hat) - (1-Y)*np.log(1-P_hat))
    Args:
        Y: labels of data
        P_hat: Estimated output probabilities from the last layer, the output layer
    Returns:
        cost: The Binary Cross-Entropy Cost result
        dP_hat: gradient of Cost w.r.t P_hat
    """
    m = Y.shape[1]  # m -> number of examples in the batch

    cost = (1 / m) * np.sum(-Y * np.log(P_hat) - (1 - Y) * np.log(1 - P_hat))
    cost = np.squeeze(cost)  # remove extraneous dimensions to give just a scalar (e.g. this turns [[17]] into 17)

    dP_hat = (1 / m) * (-(Y / P_hat) + ((1 - Y) / (1 - P_hat)))

    return cost, dP_hat


def compute_stable_bce_cost(Y, Z):
    """
    This function computes the "Stable" Binary Cross-Entropy(stable_bce) Cost and returns the Cost and its
    derivative w.r.t Z_last(the last linear node) .
    The Stable Binary Cross-Entropy Cost is defined as:
    => (1/m) * np.sum(max(Z,0) - ZY + log(1+exp(-|Z|)))
    Args:
        Y: labels of data
        Z: Values from the last linear node
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    m = Y.shape[1]

    cost = (1 / m) * np.sum(np.maximum(Z, 0) - Z * Y + np.log(1 + np.exp(- np.abs(Z))))
    dZ_last = (1 / m) * (
            (1 / (1 + np.exp(- Z))) - Y)  # from Z computes the Sigmoid so P_hat - Y, where P_hat = sigma(Z)

    return cost, dZ_last


def compute_keras_like_bce_cost(Y, P_hat, from_logits=False):
    """
    This function computes the Binary Cross-Entropy(stable_bce) Cost function the way Keras
    implements it. Accepting either probabilities(P_hat) from the sigmoid neuron or values direct
    from the linear node(Z)
    Args:
        Y: labels of data
        P_hat: Probabilities from sigmoid function
        from_logits: flag to check if logits are being provided or not(Default: False)
    Returns:
        cost: The "Stable" Binary Cross-Entropy Cost result
        dZ_last: gradient of Cost w.r.t Z_last
    """
    if from_logits:
        # assume that P_hat contains logits and not probabilities
        return compute_stable_bce_cost(Y, Z=P_hat)

    else:
        # Assume P_hat contains probabilities. So make logits out of them

        # First clip probabilities to stable range
        EPSILON = 1e-07
        P_MAX = 1 - EPSILON  # 0.9999999

        P_hat = np.clip(P_hat, a_min=EPSILON, a_max=P_MAX)

        # Second, Convert stable probabilities to logits(Z)
        Z = np.log(P_hat / (1 - P_hat))

        # now call compute_stable_bce_cost
        return compute_stable_bce_cost(Y, Z)

def generate_training_data(text, word_to_index):
    vector_data = []
    for line in text:
        vector = []
        for word in line:
            vector.append(word_to_index[word])
        vector_data.append(vector)
    return vector_data

dimension = 2
window_size = 1
learning_rate = 1
number_of_epochs = 5000
np.random.seed(48)

print('reading reviews')
data_text = get_file_data('./data/reviews.txt')
word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(data_text)
n_inputs = vocab_size
n_outputs = 2  # len(train_y[0])
data = generate_training_data(data_text, word_to_index)

weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, dimension))

weights_hidden_output = np.random.uniform(-1, 1, (dimension, vocab_size))

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
X_train = np.asarray(train_x[0:int(0.8 * len(train_x))])
Y_train = np.asarray(train_y[0:int(0.8 * len(train_y))])

print('training')

# ------ LAYER-1 ----- define output layer that takes in training data
Z1 = LinearLayer(input_shape=X_train.shape, n_out=1, ini_type='plain')
A1 = SigmoidLayer(Z1.Z.shape)

costs = []  # initially empty list, this will store all the costs after a certain number of epochs

# Start training
for epoch in range(number_of_epochs):
    # ------------------------- forward-prop -------------------------
    Z1.forward(X_train)
    A1.forward(Z1.Z)

    # ---------------------- Compute Cost ----------------------------
    cost, dZ1 = compute_stable_bce_cost(Y_train, Z1.Z)
    # print and store Costs every 100 iterations and of the last iteration.
    print("Cost at epoch#{}: {}".format(epoch, cost))
    costs.append(cost)

    # ------------------------- back-prop ----------------------------
    Z1.backward(dZ1)

    # ----------------------- Update weights and bias ----------------
    Z1.update_params(learning_rate=learning_rate)
