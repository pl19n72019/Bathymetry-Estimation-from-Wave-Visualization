# -*- coding: utf-8 -*-

import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split
from src.networks.generator import GeneratorAutoencoder


class AutoEncoder:
    """Global functionalities of the auto-encoders.

    This class treats the encoder and the decoder parts. It allows to train
    auto-encoders (define in the package `models`), and to perform all classic
    operations on the auto-encoders.

    Attributes:
        x_train (numpy.ndarray): Input/output train set.
        x_test (numpy.ndarray): Input/output test set.
        width (int): Width of the train/test data.
        height (int): Height of the train/test data.

    """

    def __init__(self, model, dataset_path=None, load_models=None, version=0):
        """Creation of the auto-encoder functionalities treatment object.

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
            load_models (str): Model to load (default: None). If it is set to
                None, no model if load. If not, it should contain a encoder
                name, and only its should be used (useful to pre-process the
                data before the input of a next neural network).
            version (int): Version of encoder to load (default: 0). If
                load_models is not None, it refers to the version of encoder to
                load. By default, it load the first model created by the outer-
                loop.

        """
        if load_models is None:
            # a new model is created, the data are load are reshaped.
            self.x_train = []
            files = glob.glob('{}/dataset/train_TS/*.npy'.format(
                '.' if dataset_path is None else dataset_path))
            for data in files:
                self.x_train.append(np.load(data)[200:])

            # reshape the data and split into two classes: train and test
            (self.width, self.height) = self.x_train[0].shape
            self.x_train = np.array(self.x_train)
            self.x_train = self.x_train.reshape((len(self.x_train), self.width,
                                                 self.height, 1))
            self.x_train, self.x_test, _, _ = train_test_split(self.x_train,
                                                               self.x_train,
                                                               test_size=0.2)

            # load the model (complete auto-encoder) and the encoder
            self.__model = model((self.width, self.height, 1))
            self.__encoder = self.__model.encoder()
            self.__model = self.__model.autoencoder()
        else:
            self.__encoder = model
            self.load_weights(load_models, full=False, version=version)

    def compile(self, optimizer='adadelta', loss='mean_squared_error'):
        """Compile the complete model.

        This method should be call only on new auto-encoders (the flag
        load_models should be set to None).

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

    def fit(self, epochs=50, repeat=1, batch_size=128, fname='autoencoder',
            fname_enc='encoder', verbose=1):
        """Trains the model for a given number of epochs (iterations on a
        dataset).

        This method should be call only on new auto-encoders (the flag
        load_models should be set to None).

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
            batch_size (int): Number of samples per gradient update
                (default: 128).
            fname (str): Name of the complete neural network on the disk
                (default: 'autoencoder').
            fname_enc (str): Name of the encoder part of the neural network on
                the disk (default: 'encoder').
            verbose (int, 0, 1, or 2): Verbosity mode (default: 1). 0 = silent,
                1 = progress bar, 2 = one line per epoch.

        Returns:
            A record of training loss values and metrics values at successive
            epochs, as well as validation loss values and validation metrics
            values (if applicable).

        """
        # saving the two architectures
        self.save_architecture(fname=fname)
        self.save_architecture(fname=fname_enc, full=False)

        # training of the model and saving of the history
        history = []
        for i in range(repeat):
            h = self.__model.fit(self.x_train,
                                 self.x_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(self.x_test, self.x_test),
                                 verbose=verbose)
            history.append(h.history)
            self.save_weights('{}.{}'.format(fname, i), architecture=False)
            self.save_weights('{}.{}'.format(fname_enc, i), full=False,
                              architecture=False)

        # the format of history is flatten
        keys = history[0].keys()
        return {k: np.array([l[k] for l in history]).flatten() for k in keys}

    def predict(self, x, batch_size=128):
        """"Generates complete output predictions for the input samples (outputs
        of the auto-encoder neural network).

        This method should be call only on new auto-encoders (the flag
        load_models should be set to None).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            x (numpy.ndarray like): Input data.
            batch_size (int): Number of samples per pass (default: 128).

        Returns:
            Numpy array(s) of reshaped predictions (for displaying).

        """
        return self.__model.predict(x, batch_size=batch_size) \
            .reshape((len(x), self.width, self.height))

    def encode(self, x, batch_size=128):
        """"Generates output predictions for the input samples (outputs of the
        encoder neural network).

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            x (numpy.ndarray like): Input data.
            batch_size (int): Number of samples per pass (default: 128).

        Returns:
            Numpy array(s) of reshaped predictions (for two-dimensional
            convolutional neural network).

        """
        _, width, height, _ = self.__encoder.output_shape
        return self.__encoder.predict(x, batch_size=batch_size) \
            .reshape((len(x), width, height, 1))

    def save_weights(self, fname='autoencoder', full=True, architecture=True):
        """Saves the weights of the networks.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None) if full is set to False. If not, this
        method should not be call.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to save (default: 'autoencoder').
            full (bool): Network flag (default: True). If `full` is set to
                True, the auto-encoder is saved and if not, only the encoder is
                saved.
            architecture (bool): Architecture flag (default: True). If
                `architecture` is set to True, the architecture the the network
                related to the flag `full` is saved in a JSON format.

        """
        file = './saves/weights/{}.h5'.format(fname)
        if full:
            self.__model.save_weights(file)
        else:
            self.__encoder.save_weights(file)

        if architecture:
            self.save_architecture(fname, full=full)

    def save_architecture(self, fname='autoencoder', full=True):
        """Saves the architectures of the networks.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None) if full is set to False. If not, this
        method should not be call.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to save (default: 'autoencoder').
            full (bool): Network flag (default: True). If `full` is set to
                True, the auto-encoder is saved and if not, only the encoder is
                saved.

        """
        file = './saves/architectures/{}.json'.format(fname)
        if full:
            with open(file, 'w') as architecture:
                architecture.write(self.__model.to_json())
        else:
            with open(file, 'w') as architecture:
                architecture.write(self.__encoder.to_json())

    def load_weights(self, fname='autoencoder', full=True, version=0):
        """Loads the weights of the networks.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None) if full is set to False. If not, this
        method should not be call.

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to save (default: 'autoencoder').
            full (bool): Network flag (default: True). If `full` is set to
                True, the auto-encoder is saved and if not, only the encoder is
                saved.
            version (int or str): Version of the network weights to load
                (default: 0). It refers to the `repeat` flag in the fitting
                method.

        """
        file = './saves/weights/{}.{}.h5'.format(fname, version)
        if full:
            self.__model.load_weights(file)
        else:
            self.__encoder.load_weights(file)

    def save_losses(self, history, fname='autoencoder'):
        """Saves the history given as input.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            history (dict): History the save on disk.
            fname (str): Name of the file to save (default: 'autoencoder').

        """
        np.save('./saves/losses/{}.npy'.format(fname),
                history)

    def load_losses(self, fname='autoencoder'):
        """Loads the history given as input.

        This method can be call on every instantiation (the flag load_models is
        not intended to be set to None).

        Note:
            This method should not be modified to be adapter to other networks.
            Only the package `models` and the main script should be modified.

        Args:
            fname (str): Name of the file to load (default: 'autoencoder').

        """
        return np.load(
            './saves/losses/{}.npy'.format(fname)).item()
