from unittest import TestCase
import numpy
from src.data_preprocessing import load_data
from src.models.convolutional_classifier import prepare_model


class TestPreprocessing(TestCase):
    def test_load_data(self):
        unique_words = 10000
        rnn_max_review_length = 100
        rnn_pad_type = trunc_type = 'pre'
        result = load_data(unique_words, rnn_max_review_length, rnn_pad_type, trunc_type, 0)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], numpy.ndarray)

class TestPrepareModel(TestCase):
    # TODO napisz jeden dowolny własny test (np. sprawdzający czy model poprawnie się stworzył)
    def test_model_creation(self):
        n_unique_words= 5000
        n_dimensions= 64
        max_text_length= 40
        drop_embed= 0.2
        kernel_num= 256
        kernel_size= 3
        n_dense= 256
        drop_dense= 0.2
        activation="relu"

        model = prepare_model(n_unique_words, n_dimensions, max_text_length, drop_embed, kernel_num, kernel_size, n_dense,
                              drop_dense, activation)

        expected_output_shape = (None, 1)
        self.assertEqual(model.output_shape, expected_output_shape)
