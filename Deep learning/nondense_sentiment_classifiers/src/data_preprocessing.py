from keras.datasets import imdb
from keras.utils import pad_sequences
import logging


def load_data(n_unique_words, max_text_length, pad_type, trunc_type, value):
    (x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)
    logging.warning("Data loaded")
    x_train = pad_sequences(x_train, maxlen=max_text_length, padding=pad_type, truncating=trunc_type, value=value)
    x_valid = pad_sequences(x_valid, maxlen=max_text_length, padding=pad_type, truncating=trunc_type, value=value)

    return x_train, y_train, x_valid, y_valid
