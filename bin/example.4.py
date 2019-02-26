#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context
import numpy as np
import matplotlib.pyplot as plt

from src.models import cnn
from src.networks.cnn import CNN

if __name__ == '__main__':
    # load an already trained cnn model
    cnn1 = CNN(load_models='cnn-Model1', version=0)


    # predict cnn
    ts_enc = np.load('dataset/train_encoded_TS/TS_00001.npy')
    width, height = ts_enc.shape
    ts_enc = np.array([ts_enc])
    bathy = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1)
    plt.subplot(1, 2, 1)
    plt.imshow(ts_enc[0])
    plt.subplot(1, 2, 2)
    plt.plot(bathy[0])
    plt.show()
