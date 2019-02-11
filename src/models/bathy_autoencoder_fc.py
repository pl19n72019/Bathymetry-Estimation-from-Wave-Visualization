from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential

class AutoEncoderModel:
    def __init__(self):
        # Define the encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(16, (3, 3), activation='sigmoid', padding='same'))
        self.encoder.add(MaxPooling2D((4, 2), padding='same'))
        self.encoder.add(Conv2D(8, (3, 3), activation='sigmoid', padding='same'))
        self.encoder.add(MaxPooling2D((2, 1), padding='same'))

        # Define the decoder
        self.decoder = Sequential()
        self.decoder.add(Conv2D(8, (3, 3), activation='sigmoid', padding='same'))
        self.decoder.add(UpSampling2D((2, 1)))
        self.decoder.add(Conv2D(16, (3, 3), activation='sigmoid', padding='same'))
        self.decoder.add(UpSampling2D((4, 2)))
        self.decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

        # Define the autoencoder
        self.autoencoder = Sequential()
        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)
