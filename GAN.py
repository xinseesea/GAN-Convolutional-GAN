import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from scipy.io import loadmat


def load_svnh():
    path = "./data/train_32x32.mat"
    data = loadmat(path)
    train_x = data['X']
    train_y = data['y']
    train_x = np.transpose(train_x, (3, 0, 1, 2))
    train_x = train_x
    print(train_x.shape)
    print(train_y.shape)
    #    for i in range(10):
    #        plt.imshow(train_x[i])
    #        plt.show()
    return train_x, train_y


train_x, train_y = load_svnh()

batch_size = 32
image_size = 28
dataset = tf.data.Dataset.from_tensor_slices(train_x).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

mnist = tf.keras.datasets.mnist
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
print(mnist_x_train.shape)
print(mnist_y_train.shape)
print(mnist_x_test.shape)
print(mnist_y_test.shape)
print(mnist_x_train[0].dtype)
mnist_dataset = tf.data.Dataset.from_tensor_slices(mnist_x_train).shuffle(1000)
mnist_dataset = mnist_dataset.batch(batch_size, drop_remainder=True).prefetch(1)

generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[image_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])

gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])


def train_gan(gan, dataset, batch_size, image_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for x_batch in dataset:
            noise = tf.random.normal(shape=[batch_size, image_size])
            generated_images = generator(noise)
            x_batch = tf.cast(x_batch, dtype=tf.float32)
            x_batch = x_batch / 255.0
            x_fake_and_real = tf.concat([generated_images, x_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(x_fake_and_real, y1)
            noise = tf.random.normal(shape=[batch_size, image_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

        noise = tf.random.normal(shape=[batch_size, image_size])
        c1 = discriminator.test_on_batch(x_fake_and_real, y1)
        c2 = gan.test_on_batch(noise, y2)
        print("discriminator, epoch:", epoch + 1, " ", c1, "\n")
        print("gan, epoch:", epoch + 1, " ", c2, "\n")


train_gan(gan, mnist_dataset, batch_size, image_size)
