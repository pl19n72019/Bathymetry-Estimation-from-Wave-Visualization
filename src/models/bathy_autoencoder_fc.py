from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential

class AutoEncoderModel:
    def __init__(self):
        # Size of the kernel to apply to convolution
        kernel_size = (5, 1)

        # Define the encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(16, kernel_size, activation='sigmoid', padding='same'))
        self.encoder.add(MaxPooling2D(2, padding='same'))
        self.encoder.add(Conv2D(8, kernel_size, activation='sigmoid', padding='same'))
        self.encoder.add(MaxPooling2D(2, padding='same'))
        self.encoder.add(Conv2D(8, kernel_size, activation='sigmoid', padding='same'))
        self.encoder.add(MaxPooling2D(2, padding='same'))

        # Define the decoder
        self.decoder = Sequential()
        self.decoder.add(Conv2D(8, kernel_size, activation='sigmoid', padding='same'))
        self.decoder.add(UpSampling2D(2))
        self.decoder.add(Conv2D(8, kernel_size, activation='sigmoid', padding='same'))
        self.decoder.add(UpSampling2D(2))
        self.decoder.add(Conv2D(16, kernel_size, activation='sigmoid', padding='same'))
        self.decoder.add(UpSampling2D(2))
        self.decoder.add(Conv2D(1, kernel_size, activation='sigmoid', padding='same'))

        # Define the autoencoder
        self.autoencoder = Sequential()
        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)
