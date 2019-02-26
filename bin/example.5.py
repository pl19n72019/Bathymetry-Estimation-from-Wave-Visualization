#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import context
import context
import numpy as np
import matplotlib.pyplot as plt

from src.models import cnn, autoencoder
from src.networks.cnn import CNN
from src.networks.autoencoder import AutoEncoder

if __name__ == '__main__':
    # load an already trained cnn model
    ae1 = AutoEncoder(load_models='encoder-Model1', version=0)
    cnn1 = CNN(load_models='cnn-Model1', version=0)


    # predict cnn, 20 predict
    ts_names = sorted(glob.glob('dataset/train_TS/*.npy'))
    bathy_names = sorted(glob.glob('dataset/train_GT/*.npy'))
    print(len(ts_names))
    print(len(bathy_names))
    for i in range(10):
        plt.subplot(4, 5, i + 1)
        
        ts_origi = np.load(ts_names[i])[200:]	# croping
        width, height = ts_origi.shape
        ts_origi = np.array([ts_origi])
        real_bathy = np.load(bathy_names[i])
        
        ts_enc = ae1.predict(ts_origi.reshape(len(ts_origi), width, height, 1), batch_size=1)
        a ,width, height = ts_enc.shape
        ts_enc = np.array([ts_enc])
        
        pred_bathy = cnn1.predict(ts_enc.reshape(len(ts_enc), width, height, 1), batch_size=1, smooth=True)
        plt.plot(real_bathy, label='Expected')
        plt.plot(pred_bathy.flatten(), label='Prediction')
        plt.legend()
    plt.show()
