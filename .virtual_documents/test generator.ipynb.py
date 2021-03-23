import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, Reshape, Conv2DTranspose
from tensorflow.nn import leaky_relu, relu, softmax, sigmoid
import numpy as np
import matplotlib.pyplot as plt


generator = Sequential()

generator.add(Dense(128*7*7, input_dim=100, activation=relu))
generator.add(Reshape((7, 7, 128)))

generator.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", activation=relu))
generator.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", activation=relu))

generator.add(Conv2D(1, (3,3), padding="same", activation=sigmoid))


generator.summary()


x_train = np.random.rand(25,100) # for 25 images with 100 latent dims


pred = generator.predict(x_train)


plt.imshow(pred[0], cmap="gray")


generator.save("generator.model")






