#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: tests_gaussian_filter.py |
Created by Benjamin on the 2019-02-05 |
Github: https://github.com/pl19n72019/bathy-vagues

Add a description.
"""


import os
import context
import numpy as np

from unittest import TestCase
from src.preprocessing.gaussian_filter import apply_gaussian_filter


class TestGaussianFilter(TestCase):

    def setUp(self):
        self.expected_output = np.load('data/gaussian_test_files/' +
                                       'file_output.npy')

    def test_correctness(self):
        path_input = 'data/gaussian_test_files/file_input.npy'
        output = apply_gaussian_filter(path_input, 1, mode='reflect')
        is_zero = np.subtract(self.expected_output, output)
        self.assertTrue(is_zero.all() == 0.)
