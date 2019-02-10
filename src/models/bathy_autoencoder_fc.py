from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, ZeroPadding1D
from keras.models import Sequential


class AutoEncoderModel:
    def __init__(self):
        # Define the encoder
        self.encoder = Sequential()
        self.encoder.add(Conv1D(16, 3, activation='relu', padding='same'))
        self.encoder.add(MaxPooling1D(2, padding='same'))
        self.encoder.add(Conv1D(8, 3, activation='relu', padding='same'))
        self.encoder.add(MaxPooling1D(2, padding='same'))
        self.encoder.add(Conv1D(8, 3, activation='relu', padding='same'))
        self.encoder.add(MaxPooling1D(2, padding='same'))

        # Define the decoder
        self.decoder = Sequential()
        self.decoder.add(Conv1D(8, 3, activation='relu', padding='same'))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(Conv1D(8, 3, activation='relu', padding='same'))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(Conv1D(16, 3, activation='relu'))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(ZeroPadding1D(padding=2))
        self.decoder.add(Conv1D(1, 3, activation='sigmoid', padding='same'))

        # Define the autoencoder
        self.autoencoder = Sequential()
        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)
