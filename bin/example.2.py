#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context
import numpy as np
import matplotlib.pyplot as plt

from src.models import autoencoder
from src.networks.autoencoder import AutoEncoder


if __name__ == '__main__':
    # load an already trained auto-encoder model
    ae1 = AutoEncoder(load_models='encoder-Model1', version=0)


    # predict encoder
    ts_origi = np.load('../dataset/train_TS/TS_00001.npy')[200:] # adapt you path
    width, height = ts_origi.shape
    ts_origi = np.array([ts_origi])
    ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
    plt.subplot(1, 2, 1)
    plt.imshow(ts_origi[0])
    plt.subplot(1, 2, 2)
    plt.imshow(ts_enc[0])
    plt.show()
