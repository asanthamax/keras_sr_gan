import keras
from keras import layers

class Discriminator:

    def __init__(self, input_dim=256, im_chan=1):
        self.input_shape = (input_dim, input_dim, im_chan+2)


    def init_disc(self):
        model = keras.models.Sequential()
        model.add(layers.InputLayer(input_shape=self.input_shape))
        model.add(layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.Conv2D(filters=512, kernel_size=4, strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same', activation='sigmoid'))
        return model