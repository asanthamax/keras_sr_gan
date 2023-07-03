import keras
from disc import Discriminator
from gen import Generator
from train_gan import SRGan
from gan_callback import GANMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import MeanSquaredError, BinaryCrossentropy

input_size=64
im_chan= 3
latent_dim = 512
disc = Discriminator(input_size, im_chan)
disc_model = disc.init_disc()
disc_model.summary()

gen = Generator(im_chan, latent_dim)
gen_model = gen.init_model()
gen_model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)

dataset = train_datagen.flow_from_directory(
    'C:\\Users\\HP\\PyCharmProjects\\SRGAN\\data',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

epochs = 100
gan = SRGan(discriminator=disc_model, generator=gen_model, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=MeanSquaredError(),
    loss_disc=BinaryCrossentropy()
)

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
)


