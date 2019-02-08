#!/usr/bin/env python
from models.bathy_autoencoder_fc import AutoEncoderModel

import os
import numpy as np
import glob

DATASET_PATH = os.path.join(os.path.abspath(__file__), "data/dataset")


def get_train_TS(nb_files):
    """Get the path of nb files for training"""
    file_listing = [f for f in glob.glob(os.path.join(DATASET_PATH, "*.npy"))]

    return file_listing[0:(nb_files-1)], file_listing[nb_files:(2*nb_files - 1)]


if __name__ == "__main__":
    # get input files
    nb_files = 10
    input_files = get_train_TS(nb_files)
    x_train, x_test = np.empty((nb_files, 600, 20, 1))
    for i, f in enumerate(input_files):
        x_train[i, ] = np.load(f)

    # get the auto encoder model
    autoencodermodel = AutoEncoderModel()
    autoencoder = autoencodermodel.get_autoencoder()

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=1,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=2)
