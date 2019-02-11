#!/usr/bin/env python
import context
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import json

from src.models.bathy_autoencoder_fc import AutoEncoderModel
from math import ceil
from random import shuffle


def get_files_TS(train_perc, test_perc):
    """Get the path of the train and test files"""
    # List all the files in the dataset directory
    file_listing = [f for f in glob.glob(os.path.join('data/dataset', "*.npy"))]
    shuffle(file_listing)

    # Compute the number of files for training
    nb_files_total = len(file_listing)
    nb_train_files = ceil(train_perc * nb_files_total)

    train_files = file_listing[0:(nb_train_files - 1)]
    test_files = file_listing[nb_train_files:]

    return train_files, test_files


if __name__ == "__main__":
    # number of time steps
    steps = 600

    # number of features in an image
    features = 200

    # number of steps that we keep (some may be useless)
    ssteps = 400

    # number of epochs
    EPOCHS = 10000

    # define the portion of training set and validation/test set
    train_perc = 0.8
    test_perc = 1.0 - train_perc

    # get input files
    train_files, test_files = get_files_TS(train_perc, test_perc)
    nb_train_files = len(train_files)
    nb_test_files = len(test_files)

    # init batches
    x_train = np.empty((nb_train_files, steps, features))
    x_test = np.empty((nb_test_files, steps, features))

    # feed batches for training set
    for i, f in enumerate(train_files):
        x_train[i, ] = np.load(f)
    x_train = x_train[:, (steps-ssteps):, :]
    x_train = np.reshape(x_train, (nb_train_files, ssteps, features, 1))

    # feed batches for test set
    for i, f in enumerate(test_files):
        x_test[i, ] = np.load(f)
    x_test = x_test[:, (steps-ssteps):, :]
    x_test = np.reshape(x_test, (nb_test_files, ssteps, features, 1))

    # get the auto encoder model
    aem = AutoEncoderModel()
    autoencoder = aem.autoencoder

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # train the model
    nb_iter = ceil(EPOCHS/50)
    for i in range(0):
        hist = autoencoder.fit(x_train, x_train,
                        epochs=50,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        verbose=2)

        # save weights
        autoencoder.save_weights(
                os.path.join('data/weights/autoencoder',
                             'autoencoder_weights_' + str(i) + '.hdf5'))

        # save history
        history_json = os.path.join('data/history/autoencoder',
                                         'autoencoder_hist_' + str(i) + '.json')
        with open(history_json, 'w') as f:
            json.dump(hist.history, f)

    test_img = autoencoder.predict(x_test)
    train_img = x_test[0, ]

    plt.figure()
    plt.imshow(np.reshape(test_img, (ssteps, features))
    plt.show()
