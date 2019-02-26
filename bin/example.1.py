#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context

from src.models import autoencoder
from src.networks.autoencoder import AutoEncoder


if __name__ == '__main__':
    # creation and training of an auto-encoder, to pre-process the data
    ae1 = AutoEncoder(model=autoencoder.Model1((400,200,1)), batch_size=4, dataset_path='.') # Adapt to your path
    ae1.compile()
    hae1 = ae1.fit(epochs=10, repeat=1, fname='autoencoder-Model1', fname_enc='encoder-Model1')
    ae1.save_losses(hae1, 'encoder-Model1') # saving the losses

    # Create the encoded dataset with the encoder
    ae1.encode_dataset('.', '.')
