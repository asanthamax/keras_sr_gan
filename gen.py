
import keras
from keras import layers

class Generator:

    def __init__(self, im_chan=1, hidden_dim=128):
        self.input_dim = hidden_dim+2
        self.output_dim = im_chan
        

    def init_model(self):
        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.input_dim,)))
        model.add(layers.Dense(7*7*self.input_dim))
        model.add(layers.Reshape((7, 7, self.input_dim)))
        model.add(layers.Conv2DTranspose(self.input_dim, kernel_size=4, strides=2, padding='same', activation='relu'))
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.Conv2DTranspose(filters=self.input_dim, kernel_size=4, strides=2, padding='same', activation='relu'))
        model.add(layers.Dropout(rate=0.5))

        model.add(layers.Conv2D(self.output_dim, kernel_size=7, padding='same', activation='sigmoid'))
        return model