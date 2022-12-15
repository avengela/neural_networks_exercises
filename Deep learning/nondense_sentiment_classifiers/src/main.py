import os
import logging

import keras.models
from keras.callbacks import ModelCheckpoint

from models import recurrent_classifier, convolutional_classifier
from data_preprocessing import load_data
from parameters import recurrent_parameters as rp
from parameters import convolutional_parameters as cp
from parameters import common_parameters as cmp
import matplotlib.pyplot as plt


def model_learning(model, name, weight_output_dir, epochs, x_train, y_train, x_valid, y_valid):
    model.compile(loss=cmp.loss, optimizer=cmp.optimizer, metrics=cmp.metrics)
    model_checkpoint = ModelCheckpoint(filepath=weight_output_dir + "/weights.{epoch:02d}.hdf5")

    if not os.path.exists(weight_output_dir):
        os.makedirs(weight_output_dir)

    model.fit(x_train, y_train, batch_size=cmp.batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid),
              callbacks=[model_checkpoint])
    model.load_weights(weight_output_dir + "/weights.03.hdf5")
    model.save(name)


# TODO edytuj funkcję results_visualisation tak, by wizualizowała wyniki dwóch modeli naraz
#  (popraw również wywołanie w mainie)
def results_visualisation(model_conv, x_valid_conv, model_rnn, x_valid_rnn):

    y_pred_conv = model_conv.predict(x_valid_conv)
    y_pred_rnn = model_rnn.predict(x_valid_rnn)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_pred_conv, bins=20, label='conv', histtype='step', fill=False)
    ax.hist(y_pred_rnn, bins=20, label='rnn', histtype='step', fill=False)
    plt.axvline(x=0.5, alpha=0.3, color='r', ls='--')
    ax.legend(loc='upper center')
    plt.show()



if __name__ == '__main__':
    model1_name = "conv_sentiment_classifier"
    model2_name = "recurrent_sentiment_classifier"

    logging.warning("Learning started")
    # TODO uzupełnij przetwarzanie tak, by przygotować dane i stworzyć modele, a następnie je naucz



    # conv
    x_train_conv, y_train_conv, x_val_conv, y_val_conv = load_data(n_unique_words=cp.conv_n_unique_words,
                                                                   max_text_length=cp.conv_max_review_length,
                                                                   pad_type=cmp.pad_type,
                                                                   trunc_type=cmp.pad_type,
                                                                   value=0.0)

    model1 = convolutional_classifier.prepare_model(n_unique_words=cp.conv_n_unique_words,
                                                    n_dimensions=cmp.n_dimensions,
                                                    max_text_length=cp.conv_max_review_length,
                                                    drop_embed=cp.dropout,
                                                    kernel_num=cp.n_conv,
                                                    kernel_size=cp.k_conv,
                                                    n_dense=cp.n_dense,
                                                    drop_dense=cp.dropout,
                                                    activation="relu"
                                                    )

    model_learning(model1,
                   model1_name,
                   weight_output_dir=cp.conv_output_dir,
                   epochs=cp.conv_epochs,
                   x_train=x_train_conv,
                   y_train=y_train_conv,
                   x_valid=x_val_conv,
                   y_valid=y_val_conv)

    # rnn
    x_train_rnn, y_train_rnn, x_val_rnn, y_val_rnn = load_data(n_unique_words=rp.rnn_n_unique_words,
                                                               max_text_length=rp.rnn_max_review_length,
                                                               pad_type=rp.rnn_pad_type,
                                                               trunc_type=rp.rnn_pad_type,
                                                               value=0.0)

    model2 = recurrent_classifier.prepare_model(n_unique_words=rp.rnn_n_unique_words,
                                                n_dimensions=cmp.n_dimensions,
                                                max_text_length=rp.rnn_max_review_length,
                                                drop_embed=rp.drop_rnn,
                                                drop_rnn=rp.drop_rnn,
                                                n_rnn=rp.n_rnn)

    model_learning(model2,
                   model2_name,
                   weight_output_dir=rp.rnn_output_dir,
                   epochs=rp.rnn_epochs,
                   x_train=x_train_rnn,
                   y_train=y_train_rnn,
                   x_valid=x_val_rnn,
                   y_valid=y_val_rnn)

    try:
        results_visualisation(model_conv=keras.models.load_model(model1_name), x_valid_conv=x_val_conv,
                              model_rnn=keras.models.load_model(model2_name), x_valid_rnn=x_val_rnn)
    except FileNotFoundError:
        logging.error("cannot find model")

    logging.warning("Learning finished")

    # TODO porównaj wyniki otrzymane przez każdy z klasyfikatorów i napisz dwa wnioski co do różnic

    conv_accuracy = keras.models.load_model(model1_name).evaluate(x_val_conv, y_val_conv)[1]
    rnn_accuracy = keras.models.load_model(model2_name).evaluate(x_val_rnn, y_val_rnn)[1]
    print(f'Conv classifier accuracy at 3rd epoch: {conv_accuracy}')
    print(f'Rnn classifier accuracy at 3rd epoch: {rnn_accuracy}')

    #Model wykorzystujący recurrent classifier uczy się wolniej. Przy trzeciej epoce jego accuracy
    #jest dużo mniejsze od accuracy drugiego modelu.

    #Model wykorzystujący convolutional classifier zyskał większą pewność od drugiego modelu.