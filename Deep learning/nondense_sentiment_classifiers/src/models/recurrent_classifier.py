from keras.models import Sequential
from keras.layers import SpatialDropout1D, SimpleRNN, Embedding, Dense


def prepare_model(n_unique_words, n_dimensions, max_text_length, drop_embed, n_rnn, drop_rnn):
    # TODO odpowiednio osadź przestrzeń wektorową (embedding), dodaj przestrzenny dropout z odpowiednim parametrem
    #  i wybraną funkcją aktywacji, utwórz prostą warstwę rekurencyjną (SimpleRNN) z odpowiednimi hiperparametrami
    #  zakończ warstwą klasyfikującą


    model = Sequential()

    model.add(Embedding(input_dim=n_unique_words, output_dim=n_dimensions, input_length=max_text_length))
    model.add(SpatialDropout1D(drop_embed))
    model.add(SimpleRNN(n_rnn, return_sequences=False, activation='relu', dropout=drop_rnn))
    model.add(Dense(1, activation="sigmoid"))

    return model
