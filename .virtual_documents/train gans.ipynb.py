import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


generator = load_model("generator.model")


discriminator = load_model("discriminator.model")
discriminator.trainable = False


gans = Sequential([generator, discriminator])
gans.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

for counter in tqdm(range(1, 10)):
    x_train = np.random.rand(1000, 100)
    y_train = np.ones(1000)
    gans.fit(x_train, y_train, epochs=1, batch_size=32)
    if counter % 5 == 0:
        gans.save("gans.model")


gans.save("gans.model")



