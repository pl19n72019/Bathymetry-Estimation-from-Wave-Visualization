#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context
import matplotlib.pyplot as plt

from src.models import cnn, autoencoder
from src.networks.autoencoder import AutoEncoder
from src.networks.cnn import CNN

if __name__ == '__main__':
    # creation and training of an auto-encoder, to pre-process the data
    #ae1 = AutoEncoder(model=autoencoder.Model1((400,200,1)), batch_size=128, dataset_path='..')
    #ae1.compile()
    hae1 = ae1.fit(epochs=10, repeat=1, fname='autoencoder-Model1', fname_enc='encoder-Model1')

    # saving of the computed auto-encoder metrics (complete network)
    #ae1.save_losses(hae1, 'encoder-Model1')

    #ae1.encode_dataset('../dataset/train_TS', '../dataset/train_encoded_TS')



    ae1 = AutoEncoder(load_models='encoder-Model1')
    ae1.encode_dataset('../dataset/train_TS', '../dataset/train_encoded_TS')




    # printing of the different loss of the training
    # plt.plot(hae1['loss'], label='Training')
    # plt.plot(hae1['val_loss'], label='Validation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Auto-encoder')
    # plt.legend()
    # plt.show()

    # printing of the first test (input and output of the auto-encoder)
    # p = ae1.predict(ae1.x_test, __batch_size=8)
    # (w, h, _) = ae1.x_test[0].shape  # ae1.x_test[0].shape == (w, h, 1)
    # plt.subplot(121)
    # plt.imshow(ae1.x_test[0].reshape(w, h))
    # plt.title('Expected')
    # plt.subplot(122)
    # plt.imshow(p[0])
    # plt.title('Prediction')
    # plt.show()

    # creation and training of the CNN, to process the data. the pre-processing
    # of the data is done using the previous auto-encoder
    #cnn1 = CNN(model=cnn.Model1((100,200,1), 200), batch_size=8, dataset_path='..')
    #cnn1.compile()
    #hcnn1 = cnn1.fit(epochs=10, repeat=1, fname='cnn-Model1')

    # saving of the computed network metrics (only the CNN part)
    #cnn1.save_losses(hcnn1, 'cnn-Model1')

    # printing of the different loss of the training
    # plt.plot(hcnn1['loss'], label='Training')
    # plt.plot(hcnn1['val_loss'], label='Validation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Convolutional network')
    # plt.legend()
    # plt.show()

    # printing of some examples of bathymetry prediction
    #p = cnn1.predict(cnn1.x_test, batch_size=8)
    #for i in range(min(20, len(p))):
    #    plt.subplot(4, 5, i + 1)
    #    plt.plot(cnn1.y_test[i], label='Expected')
    #    plt.plot(p[i], label='Prediction')
    #    plt.legend()
    #plt.show()

    #p = cnn1.predict(cnn1.x_test, batch_size=8, smooth=True)
    #for i in range(min(20, len(p))):
    #    plt.subplot(4, 5, i + 1)
    #    plt.plot(cnn1.y_test[i], label='Expected')
    #    plt.plot(p[i], label='Prediction')
    #    plt.legend()
    #plt.show()
