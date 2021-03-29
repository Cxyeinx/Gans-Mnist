import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2DTranspose, Conv2D, BatchNormalization, Reshape, Dropout
from tensorflow.nn import relu, tanh, leaky_relu, sigmoid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train / 255
x_test = x_test / 255
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


generator = Sequential()

generator.add(Dense(128*7*7, input_dim=100, activation=relu))
generator.add(Reshape((7, 7, 128)))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", activation=relu))
generator.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", activation=relu))
generator.add(BatchNormalization())
generator.add(Dropout(0.3))

generator.add(Conv2D(64, (3,3), padding="same", activation=relu))
generator.add(Conv2D(1, (3,3), padding="same", activation=tanh))

generator.summary()


discriminator = Sequential()

discriminator.add(Conv2D(64, (3,3), padding="same", input_shape=(28,28,1), activation=leaky_relu))
discriminator.add(BatchNormalization())

discriminator.add(Conv2D(128, (3,3), padding="same", activation=leaky_relu))
discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(64, (3,3), padding="same", activation=leaky_relu))
discriminator.add(BatchNormalization())
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(64, activation=leaky_relu))
discriminator.add(Dense(128, activation=leaky_relu))
discriminator.add(Dense(1, activation=sigmoid))

discriminator.compile(loss ="binary_crossentropy", optimizer ="adam")
discriminator.trainable = False
discriminator.summary()


gan = Sequential([generator, discriminator])

gan.compile(loss ="binary_crossentropy", optimizer ="adam")


gen, dis = gan.layers


for epoch in tqdm(range(100)):
#     real samples for discriminator
    if epoch <= 1:
        random = np.random.randint(0, 30_000)
        x_real = x_train[0:random]
        np.random.shuffle(x_real)
        y_real = np.ones(random)

        # fake samples for discriminator
        x_fake = np.random.rand(random, 28, 28, 1)
        y_fake = np.zeros(random)

        # make a huge array for discriminator
        x = np.concatenate((x_real, x_fake))
        y = np.concatenate((y_real, y_fake))

        # train the discriminator
        dis.trainable = True
        dis.fit(x, y, epochs=1, verbose=0)

    #train the gans model
    noise = np.random.rand(100,100)
    y2 = np.ones(100)
    dis.trainable = False
    gan.fit(noise, y2, epochs=5, verbose=0)
    
    # save the prediction
    pred = gen(np.random.rand(100,100), training=False)
    pred = np.array(pred[33]).reshape(28,28)
    plt.imsave(f"{epoch}.png", pred)










