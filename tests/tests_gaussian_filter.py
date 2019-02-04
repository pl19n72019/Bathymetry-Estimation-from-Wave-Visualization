from unittest import TestCase
from preprocessing.gaussian_filter import apply_gaussian_filter

import os
import numpy as np


FILE_PATH = os.path.dirname(__file__)


class TestGaussianFilter(TestCase):

    def setUp(self):
        self.expected_output = np.load(os.path.join(FILE_PATH, 'test_files/gaussian_test_files/file_output.npy'))

    def test_correctness(self):
        path_input = os.path.join(FILE_PATH, 'test_files/gaussian_test_files/file_input.npy')
        output = apply_gaussian_filter(path_input, 1, mode='reflect')
        is_zero = np.subtract(self.expected_output, output)
        self.assertTrue(is_zero.all() == 0.)
