import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow
from tensorflow.keras.datasets.mnist import load_data
import numpy as np


# discriminator to check that the generated is even readable or not :P
discriminator = Sequential()

discriminator.add(Conv2D(64, (3,3), padding="same", input_shape=(28,28,1), activation=leaky_relu))
discriminator.add(Dropout(0.3))
discriminator.add(BatchNormalization())

discriminator.add(Conv2D(128, (3,3), padding="same", activation=leaky_relu))
discriminator.add(Dropout(0.3))
discriminator.add(BatchNormalization())

discriminator.add(Conv2D(64, (3,3), padding="same", activation=leaky_relu))
discriminator.add(Dropout(0.3))
discriminator.add(BatchNormalization())

discriminator.add(Flatten())
discriminator.add(Dense(64, activation=leaky_relu))
discriminator.add(Dense(128, activation=leaky_relu))
discriminator.add(Dense(1, activation=sigmoid))

discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


(x_train_real, _), (_, _) = load_data()
x_train_real = x_train_real / 255
x_train_real = np.expand_dims(x_train_real, axis=-1)
y_train_real = np.ones(len(x_train_real))


x_train_fake = np.random.rand(60_000, 28, 28, 1)
y_train_fake = np.zeros(60_000)


print(x_train_real.shape, y_train_real.shape, x_train_fake.shape, y_train_fake.shape)


print(y_train_real)
print(y_train_fake)


x_train = np.concatenate((x_train_real, x_train_fake))
y_train = np.concatenate((y_train_real, y_train_fake))


print(x_train.shape, y_train.shape)


model.fit(x_train, y_train, epochs=1, shuffle=True, validation_split=0.2)


model.save("discriminator.model")



