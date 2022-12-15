from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import SpatialDropout1D, Conv1D, GlobalMaxPooling1D


def prepare_model(n_unique_words, n_dimensions, max_text_length, drop_embed, kernel_num, kernel_size, n_dense,
                  drop_dense, activation):
    # TODO odpowiednio osadź przestrzeń wektorową (embedding), dodaj prdrop_embedzestrzenny dropout z odpowiednim parametrem
    #  i wybraną funkcją aktywacji, utwórz warstwę konwolucyjną jednowymiarową z odpowiednimi hiperparametrami,
    #  dodaj warstwę redukującą - globalny jednowymiarowy max-pooling, uzupełnij sieć o warstwę gęstą z wybraną
    #  funkcją aktywacji i dropoutem, zakończ warstwą klasyfikującą

    model = Sequential()

    model.add(Embedding(input_dim = n_unique_words, output_dim = n_dimensions, input_length = max_text_length))
    model.add(SpatialDropout1D(drop_embed))
    model.add(Conv1D(filters=kernel_num, kernel_size=kernel_size, padding='same', activation=activation))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(n_dense, activation=activation))
    model.add(Dropout(rate=drop_dense))
    model.add(Dense(1, activation="sigmoid"))

    return model
