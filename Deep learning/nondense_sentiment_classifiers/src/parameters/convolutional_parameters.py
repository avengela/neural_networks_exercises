# lokalizacja wyjściowa dla wag:
conv_output_dir = 'model_output/conv'

# parametry treningu:
conv_epochs = 10

# osadzenie przestrzeni wektorowej:
conv_n_unique_words = 5000
conv_max_review_length = 400

# parametry architektury konwolucyjnej:
n_conv = 256  # filtry/kernele
k_conv = 3  # długość kernela

# parametry warstwy gęstej:
n_dense = 256 # liczba neuronów
dropout = 0.2
