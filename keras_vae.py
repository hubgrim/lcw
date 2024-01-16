import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from scipy.spatial.distance import cdist
import math

def silverman_rule_of_thumb_normal(N: int) -> float:
    return (4/(3*N))**0.4

def pairwise_distances(x, y):
    if y is None:
        y = x
    x_np = x.numpy()
    y_np = y.numpy()
    distances_np = cdist(x, y)**2
    distances_tf = tf.constant(distances_np)
    return distances_tf

def euclidean_norm_squared(X, axis):
    return K.sum(K.square(K.sqrt(K.sum(K.square(X), axis=axis))), axis=-1)

def cw_normality(X, y = None):
    assert len(X.size()) == 2

    N, D = X.size()

    if y is None:
        y = silverman_rule_of_thumb_normal(N)

    K1 = 1.0/(2.0*D-3.0)

    A1 = pairwise_distances(X)
    A = tf.reduce_mean(1/K.sqrt(y + K1*A1))

    B1 = euclidean_norm_squared(X, axis=1)
    B = 2* tf.reduce_mean((1/K.sqrt(y + 0.5 + K1*B1)))

    return (1/math.sqrt(1+y)) + A - B

def phi_sampling(s, D):
    return (1.0 + 4.0*s/(2.0*D-3))**(-0.5)

def cw_sampling(first_sample, second_sample, y):
    shape = first_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    first_sample = tf.reshape(first_sample, [-1, dim])
    
    shape = second_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    second_sample = tf.reshape(second_sample, [-1, dim])

    assert len(first_sample.size()) == 2
    assert first_sample.size() == second_sample.size()

    _, D = first_sample.size()

    T = 1.0/(2.0*math.sqrt(math.pi*y))

    A0 = pairwise_distances(first_sample)
    A = (phi_sampling(A0/(4*y), D)).mean()

    B0 = pairwise_distances(second_sample)
    B = (phi_sampling(B0/(4*y), D)).mean()

    C0 = pairwise_distances(first_sample, second_sample)
    C = (phi_sampling(C0/(4*y), D)).mean()

    return T*(A + B - 2*C)

def cw_sampling_silverman(first_sample, second_sample):
    with tf.no_grad():
        stddev = tf.math.reduce_std(second_sample)
        N = tf.shape(second_sample)[0]
        gamma = silverman_rule_of_thumb_normal(N)
    return cw_sampling(first_sample, second_sample, gamma)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)

