#!/usr/bin/env python
from models.bathy_autoencoder_fc import AutoEncoderModel

import os
import numpy as np
import glob
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join(
        os.path.dirname(__file__),
        "data/dataset")


def get_train_TS(nb_files):
    """Get the path of nb files for training"""
    file_listing = [f for f in glob.glob(os.path.join(DATASET_PATH, "*.npy"))]

    return file_listing[0:(nb_files-1)], file_listing[nb_files:(2*nb_files-1)]


if __name__ == "__main__":
    # get input files
    nb_files = 10
    train_files, test_files = get_train_TS(nb_files)
    x_train = np.empty((nb_files, 600, 200))
    x_test = np.empty((nb_files, 600, 200))
    print(train_files)
    print(DATASET_PATH)
    for i, f in enumerate(train_files):
        x_train[i, ] = np.load(f)
    x_train = np.reshape(x_train, (len(x_train), 600, 200, 1))

    for i, f in enumerate(test_files):
        x_test[i, ] = np.load(f)
    x_test = np.reshape(x_test, (len(x_test), 600, 200, 1))

    # get the auto encoder model
    autoencodermodel = AutoEncoderModel()
    autoencoder = autoencodermodel.autoencoder

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=1,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=2)

    test_img = autoencoder.predict(np.reshape(x_test[1, ], (1, 600, 200, 1)))
    train_img = x_test[1, ]

    train_img = (train_img - max(train_img))/(max(train_img) - min(train_img))
    test_img = (test_img - max(test_img))/(max(test_img) - min(test_img))

    plt.figure()
    plt.imshow(train_img)
    plt.show()

    plt.figure()
    plt.imshow(test_img)
    plt.show()
