from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


class AutoEncoderModel:
    def __init__(self):
        self.input_img = Input(shape=(600, 200, 1))
        self.input_encoded = Input(shape=self.get_encoded().output_shape)
        self.encoded = self.get_encoded()
        self.encoder = Model(self.input_img, self.encoded)
        self.decoder = Model(self.input_encoded,
                             self.get_decoded(self.input_encoded))

    def get_encoded(self):
        """Get the encoded layer"""

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(
            self.input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        return encoded

    def get_decoded(self, encoded):
        """Get the decoded layer"""

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        return decoded

    def get_autoencoder(self):
        """Get the auto encoder"""

        return Model(self.input_encoded, self.get_decoded(self.encoded))
