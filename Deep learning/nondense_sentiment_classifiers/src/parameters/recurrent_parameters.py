# lokalizacja wyjściowa dla wag:
rnn_output_dir = 'model_output/rnn'

# parametry treningu:
rnn_epochs = 4
rnn_batch_size = 128

# osadzenie przestrzeni wektorowej:
rnn_n_unique_words = 10000
rnn_max_review_length = 100  # zmniejszone względem conv z powodu szybciej zanikającego gradientu w jednostkach rnn
rnn_pad_type = trunc_type = 'pre'

# parametry architektury rekurencyjnej:
n_rnn = 256 # liczba neuronów rekurencyjnych
drop_rnn = 0.2 # dropout na warstwie rekurencyjnej
