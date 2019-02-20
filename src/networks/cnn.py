# -*- coding: utf-8 -*-

import glob
import os

import numpy as np
from keras.models import model_from_json
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

from src.networks.autoencoder import AutoEncoder
from src.networks.generator import GeneratorCNN


class CNN:
    """Global functionalities of the convolutional neural networks.

    This class  allows to train convolutional network network (define in the
    package `models`), and to perform all classic operations on the neural
    networks.

    Attributes:

    """

    def __init__(self, model, dataset_path=None, batch_size=128):
        """Creation of the CNN functionalities treatment object.

        It loads and create the dataset and the models. Only 80% of the load
        data are used for the training phase. The other 20% are use to validate
        the model during the fitting phase. Moreover, the shape of the data
        (train and test) are fitting to run on 2-dimensional convolutional
        neural network (which offers the best results). If other network are
        used, reshape the data at the input of the network.

        Note:
            This file should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            model: Model of auto-encoder (should be created in the package
                `models`).
            dataset_path (str): Path to the dataset (default: None). If it is
                set to None, the current path is selected.
            batch_size (int): Size of the batch.
        """
        self.batch_size = batch_size
        # load the model
        self.__model = model.model()

        # creation of the generator
        ts_files = sorted(glob.glob('{}/dataset/train_encoded_TS/*.npy'.format(
            '.' if dataset_path is None else dataset_path)))
        gt_files = sorted(glob.glob('{}/dataset/train_GT/*.npy'.format(
            '.' if dataset_path is None else dataset_path)))

        ts_train, ts_test, b_train, b_test = train_test_split(ts_files, gt_files,
                                                              test_size=0.2)

        self._nb_train_samp = len(ts_train)
        self._nb_test_samp = len(ts_test)
        self._generator_train = GeneratorCNN(ts_train, b_train, self.batch_size)
        self._generator_test = GeneratorCNN(ts_test, b_test, self.batch_size)

    def compile(self, optimizer='adadelta', loss='mean_squared_error'):
        """Compile the complete model.

        This method should be call only on new neural network.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            optimizer (str (name of optimizer) or optimizer instance): Optimizer
                used to train the neural network (default: 'adadelta').
            loss (str (name of objective function) ob objective function):
                Objective function used to analyze the network during the
                different phases (default: 'mean_squared_error').

        """
        self.__model.compile(optimizer=optimizer, loss=loss)

    def fit(self, epochs=50, repeat=1, fname='cnn', verbose=1):
        """Trains the model for a given number of epochs (iterations on a
        dataset).

        This method should be call only on new neural network.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            epochs (int): Number of epochs to run on each outer-loop (default:
                50). At the end of each epoch iterations, the neural networks
                are saved.
            repeat (int): Number of outer-loops (default: 1). At the end of the
                training phase, `epochs * repeats` epochs are performed.
                Moreover, `repeat` networks are saved on the disk.
            fname (str): Name of the complete neural network on the disk
                (default: 'cnn').
            verbose (int, 0, 1, or 2): Verbosity mode (default: 1). 0 = silent,
                1 = progress bar, 2 = one line per epoch.

        Returns:
            A record of training loss values and metrics values at successive
            epochs, as well as validation loss values and validation metrics
            values (if applicable).

        """
        # saving the two architectures
        self.save_architecture(fname=fname)

        # training of the model and saving of the history
        history = []
        for i in range(repeat):
            h = self.__model.fit_generator(generator=self._generator_train,
                                           steps_per_epoch=(self._nb_train_samp//self.batch_size),
                                           epochs=epochs,
                                           verbose=verbose,
                                           validation_data=self._generator_test,
                                           validation_steps=(self._nb_test_samp//self.batch_size))
            history.append(h.history)
            self.save_weights('{}.{}'.format(fname, i), architecture=False)

        # the format of history is flatten
        keys = history[0].keys()
        return {k: np.array([l[k] for l in history]).flatten() for k in keys}

    def predict(self, x, batch_size=128, smooth=False, smooth_args=(53, 2)):
        """Generates complete predictions for the input samples.

        This method should be call only on new neural network.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            x (numpy.ndarray like): Input data.
            batch_size (int): Number of samples per pass (default: 128).
            smooth (bool): Smoothing flag (default: False). If it is set to
                True, the output is smoothed using the `smooth_args` parameters.
            smooth_args (tuple): Smooth algorithm parameters (default: (53, 2)).

        Returns:
            Numpy array(s) of reshaped predictions (for displaying).

        """
        prediction = self.__model.predict(x, batch_size=batch_size) \
            .reshape((len(x), self.output_size))

        # smoothing using the Savitzky-Golay filter
        if smooth:
            sp = []
            for p in prediction:
                sp.append(savgol_filter(p, smooth_args[0], smooth_args[1]))
            prediction = np.array(sp)

        return prediction

    def save_weights(self, fname='cnn', architecture=True):
        """Saves the weights of the networks.

        This method can be call on every instantiation.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to save (default: 'cnn').
            architecture (bool): Architecture flag (default: True). If
                `architecture` is set to True, the architecture the the network
                related to the flag `full` is saved in a JSON format.

        """
        self.__model.save_weights(
            './saves/weights/{}.h5'.format(fname))

        if architecture:
            self.save_architecture(fname)

    def save_architecture(self, fname='cnn'):
        """Saves the architectures of the networks.

        This method can be call on every instantiation.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to save (default: 'cnn').

        """
        with open('./saves/architectures/{}.json'.format(fname),
                  'w') as architecture:
            architecture.write(self.__model.to_json())

    def load_weights(self, fname='cnn'):
        """Loads the weights of the networks.

        This method can be call on every instantiation.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to save (default: 'cnn').

        """
        self.__model.load_weights(
            './saves/weights/{}.h5'.format(fname))

    def save_losses(self, history, fname='cnn'):
        """Saves the history given as input.

        This method can be call on every instantiation.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            history (dict): History the save on disk.
            fname (str): Name of the file to save (default: 'cnn').

        """
        np.save('./saves/losses/{}.npy'.format(fname),
                history)

    def load_losses(self, fname='cnn'):
        """Loads the history given as input.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to load (default: 'cnn').

        """
        return np.load(
            './saves/losses/{}.npy'.format(fname)).item()
