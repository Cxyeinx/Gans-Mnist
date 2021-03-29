import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Conv2DTranspose, BatchNormalization, Reshape, Dropout
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.nn import tanh, leaky_relu, sigmoid


(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
x_train.shape, x_test.shape


plt.figure(figsize = (10, 10))
for i in range(1, 26):
    plt.subplot(5, 5, i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap = plt.cm.binary)
plt.show()


BATCH_SIZE = 32
NOISE = 100
SEED = np.random.rand(BATCH_SIZE, 100)


def create_batch(x_train):
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(BATCH_SIZE)
    return dataset


generator = Sequential()

generator.add(Dense(7 * 7 * 128, input_shape =[NOISE]))
generator.add(Reshape((7, 7, 128)))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding ="same", activation =leaky_relu))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(1, (3, 3), strides=(2, 2), padding ="same", activation ="tanh"))

generator.summary()


discriminator = Sequential()

discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding ="same", input_shape =(28, 28, 1), activation=leaky_relu))
discriminator.add(Dropout(0.3))

discriminator.add(Conv2D(128, (3, 3), strides=(2, 2), padding ="same", activation=leaky_relu))
discriminator.add(Dropout(0.3))

discriminator.add(Flatten())
discriminator.add(Dense(1, activation =sigmoid))

discriminator.summary()


discriminator.compile(loss ="binary_crossentropy", optimizer ="adam")
discriminator.trainable = False

gans = Sequential([generator, discriminator])
gans.compile(loss ="binary_crossentropy", optimizer ="adam")


def train_dcgan(gan, dataset):
    generator, discriminator = gan.layers
    epochs = 100
    index = 0
    for epoch in tqdm(range(epochs)):

        for x_batch in dataset:
            if x_batch.shape == (32, 28, 28, 1):
                noise = np.random.rand(BATCH_SIZE, NOISE)
                generated_images = generator(noise)

                x_fake_and_real = np.concatenate((generated_images, x_batch))
                y1 = np.concatenate(( np.zeros(BATCH_SIZE), np.ones(BATCH_SIZE) ))
                discriminator.trainable = True
                discriminator.train_on_batch(x_fake_and_real, y1)

                # Here we will be training our GAN model, in this step
                #  we pass noise that uses geeneratortogenerate the image
                #  and pass it with labels as [1] So, it can fool the discriminatoe
                noise = np.random.rand(BATCH_SIZE, NOISE)
                y2 = np.ones(BATCH_SIZE)
                discriminator.trainable = False
                gan.train_on_batch(noise, y2)
        
        index += 1
        # generate images for the GIF as we go
        if index % 10 == 0:
            generate_and_save_images(generator, index+1, SEED)
        
        



def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)

    fig = plt.figure(figsize =(10, 10))
  
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i], cmap ='binary')
        plt.axis('off')
    
    plt.savefig(f"./images/{epoch}.png")


dataset = create_batch(x_train[:30_000])
train_dcgan(gans, dataset)



