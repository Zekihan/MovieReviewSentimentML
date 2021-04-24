import tensorflow as tf
print(tf.__version__)
import seaborn as sns
sns.set()

from tensorflow.python.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])