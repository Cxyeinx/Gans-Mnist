import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("discriminator.model")


x = np.random.rand(10,28,28,1)
pred = model.predict(x)
pred



