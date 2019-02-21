#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context

from src.models import cnn
from src.networks.cnn import CNN

if __name__ == '__main__':
    # creation and training of the CNN, to process the data. the pre-processing
    # of the data is done using the previous encoded dataset
    cnn1 = CNN(model=cnn.Model1((100,200,1), 200), batch_size=8, dataset_path='..') # Adapt to your path
    cnn1.compile()
    hcnn1 = cnn1.fit(epochs=10, repeat=1, fname='cnn-Model1')

    # saving of the computed network metrics (only the CNN part)
    cnn1.save_losses(hcnn1, 'cnn-Model1')
