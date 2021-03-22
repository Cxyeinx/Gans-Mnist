import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, Reshape, Conv2DTranspose
from tensorflow.nn import leaky_relu, relu, softmax, sigmoid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


gans = load_model("gans.model")


for i in tqdm(range(1000)):
    x_train = np.random.rand(1000,100)
    y_train = np.ones(1000)
    gans.train_on_batch(x_train, y_train)


gans.save("gans.model")


x_train = np.random.rand(256,100)
y_train = np.ones(256)
loss, accuracy = gans.evaluate(x_train, y_train)


gans.summary()


from tensorflow.keras import backend as K
get_3rd_layer_output = K.function([gans.layers[0].input],
                                  [gans.layers[0].output])
layer_output = get_3rd_layer_output([x_train])[0]


plt.imshow(layer_output[100], cmap="gray")






