import torch
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from scipy.spatial.distance import cdist
import math


#
# def pairwise_distances_tf(x, y=None):
#     if y is None:
#         y = x
#     x_np = x.numpy()
#     y_np = y.numpy()
#
#     # euclidean distance squared
#     distances_np = cdist(x_np, y_np) ** 2
#
#     distances_tf = tf.constant(distances_np)
#     return distances_tf
#
#
# def pairwise_distances_tf2(x, y=None):
#     if y is None:
#         y = x
#     distances_tf = tf.norm(x[:, None] - y, axis=-1) ** 2
#     return distances_tf
#
#
# def pairwise_distances_torch(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
#     if y is None:
#         y = x
#     return torch.cdist(x, y) ** 2
#
#
# def test_distances():
#     tf_tensor = tf.constant([[1., 1., 1.], [3., 2., 1.]])
#     torch_tensor = torch.tensor([[1., 1., 1.], [3., 2., 1.]])
#     pairwise_distances_tensorflow = pairwise_distances_tf(tf_tensor)
#     pairwise_distances_pytorch = pairwise_distances_torch(torch_tensor)
#     pairwise_distances_tensorflow2 = pairwise_distances_tf2(tf_tensor)
#     torch_n = torch_tensor.size(0)
#     tf_n = tf.shape(tf_tensor)[0]
#
#     print("done")

def silverman_rule_of_thumb_normal(N):
    return tf.pow((4 / (3 * N)), 0.4)


def pairwise_distances(x, y=None):
    if y is None:
        y = x
    distances_tf = tf.norm(x[:, None] - y, axis=-1) ** 2
    return tf.cast(distances_tf, dtype=tf.float64)


def cw_normality(X, y=None):
    assert len(X.shape) == 2

    D = tf.cast(tf.shape(X)[1], tf.float64)
    N = tf.cast(tf.shape(X)[0], tf.float64)

    if y is None:
        y = silverman_rule_of_thumb_normal(N)

    # adjusts for dimensionality; D=2 -> K1=1, D>2 -> K1<1
    K1 = 1.0 / (2.0 * D - 3.0)

    A1 = pairwise_distances(X)
    A = tf.reduce_mean(1 / tf.math.sqrt(y + K1 * A1))

    B1 = tf.cast(tf.square(tf.math.reduce_euclidean_norm(X, axis=1)), dtype=tf.float64)
    B = 2 * tf.reduce_mean((1 / tf.math.sqrt(y + 0.5 + K1 * B1)))

    return (1 / tf.sqrt(1 + y)) + A - B


def phi_sampling(s, D):
    return tf.pow(1.0 + 4.0 * s / (2.0 * D - 3), -0.5)


def cw_sampling_lcw(first_sample, second_sample, y):
    shape = first_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    first_sample = tf.reshape(first_sample, [-1, dim])

    shape = second_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    second_sample = tf.reshape(second_sample, [-1, dim])

    assert len(first_sample.shape) == 2
    assert first_sample.shape == second_sample.shape

    _, D = first_sample.shape

    T = 1.0 / (2.0 * tf.sqrt(math.pi * y))

    A0 = pairwise_distances(first_sample)
    A = tf.reduce_mean(phi_sampling(A0 / (4 * y), D))

    B0 = pairwise_distances(second_sample)
    B = tf.reduce_mean(phi_sampling(B0 / (4 * y), D))

    C0 = pairwise_distances(first_sample, second_sample)
    C = tf.reduce_mean(phi_sampling(C0 / (4 * y), D))

    return T * (A + B - 2 * C)


def euclidean_norm_squared(X, axis=None):
    return tf.reduce_sum(tf.square(X), axis=axis)


def cw_sampling_silverman(first_sample, second_sample):
    stddev = tf.math.reduce_std(second_sample)
    N = tf.cast(tf.shape(second_sample)[0], tf.float64)
    gamma = silverman_rule_of_thumb_normal(N)
    return cw_sampling_lcw(first_sample, second_sample, gamma)


@tf.keras.saving.register_keras_serializable()
class Encoder(keras.Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.activation = layers.Activation("relu")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256)
        self.dense2 = layers.Dense(args["latent_dim"], name="z")

    def build(self, **kwargs):
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = self.flatten(encoder_inputs)
        x = self.dense1(x)
        x = self.activation(x)
        z = self.dense2(x)
        encoder = keras.Model(encoder_inputs, [z], name="encoder")
        return encoder


@tf.keras.saving.register_keras_serializable()
class Decoder(keras.Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = args["latent_dim"]
        self.activation = layers.Activation("relu")
        self.dense1 = layers.Dense(256)
        self.dense2 = layers.Dense(28 * 28, activation="sigmoid")
        self.reshape = layers.Reshape([28, 28, 1])

    def build(self, **kwargs):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = self.dense1(latent_inputs)
        x = self.activation(x)
        x = self.dense2(x)
        decoder_outputs = self.reshape(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="encoder")
        return decoder


@tf.keras.saving.register_keras_serializable()
class Generator(keras.Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.noise_dim = args["noise_dim"]
        self.activation = layers.Activation("relu")
        self.dense1 = layers.Dense(512)
        self.dense2 = layers.Dense(args["latent_dim"], name="z")

    def build(self, **kwargs):
        noise_inputs = keras.Input(shape=(self.noise_dim,))
        x = self.dense1(noise_inputs)
        x = self.activation(x)
        z = self.dense2(x)
        latent_generator = keras.Model(noise_inputs, [z], name="generator")
        return latent_generator


@tf.keras.saving.register_keras_serializable()
class BaseNet(keras.Model):
    def __init__(self, args, **kwargs):
        super(BaseNet, self).__init__(**kwargs)
        self.encoder = Encoder(args).build()
        self.decoder = Decoder(args).build()
        self.args = args
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="cw_reconstruction_loss"
        )
        self.cw_loss_tracker = keras.metrics.Mean(name="cw_loss")

    def get_config(self):
        config = {
            "args": self.args
        }
        base_config = super(BaseNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.cw_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            # tf.print(reconstruction)
            cw_reconstruction_loss = tf.math.log(
                cw_sampling_silverman(data, reconstruction))
            lambda_val = 1
            cw_loss = lambda_val * tf.math.log(cw_normality(z))
            total_loss = cw_reconstruction_loss + cw_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(cw_reconstruction_loss)
        self.cw_loss_tracker.update_state(cw_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "cw_reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "cw_loss": self.cw_loss_tracker.result(),
        }


@tf.keras.saving.register_keras_serializable()
class HighNet(keras.Model):
    def __init__(self, encoder, decoder, args, **kwargs):
        super(HighNet, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.args = args
        self.generator = Generator(args).build()
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="cw_reconstruction_loss"
        )

    def get_config(self):
        config = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "args": self.args
        }
        base_config = super(HighNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        x = self.encoder(inputs)
        return self.decoder(x)

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            batch_size = tf.shape(z)[0]
            noise_np = np.random.normal(0, 1, size=self.args["noise_dim"])
            noise_tf = tf.expand_dims(tf.convert_to_tensor(noise_np), axis=0)
            noise_tf = tf.repeat(noise_tf, repeats=batch_size, axis=0)
            noise_z = self.generator(noise_tf)
            # tf.print(reconstruction)
            cw_reconstruction_loss = tf.math.log(
                cw_sampling_silverman(z, noise_z))
        grads = tape.gradient(cw_reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(cw_reconstruction_loss)
        return {
            "cw_reconstruction_loss": self.reconstruction_loss_tracker.result()
        }


def test_saving():
    args = {"sample_amount": 1000,
            "latent_dim": 24,
            "noise_dim": 24,
            "epochs": 1,
            "batch_size": 128,
            "patience": 3,
            "learning_rate": 0.0001}
    (x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)[0:100]
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    base_model = BaseNet(args)
    base_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args["learning_rate"]))
    es_callback = keras.callbacks.EarlyStopping(monitor='total_loss', patience=args["patience"], mode="min")
    base_model.fit(mnist_digits, epochs=args["epochs"], batch_size=args["batch_size"], callbacks=[es_callback])

    model = HighNet(base_model.encoder, base_model.decoder, args)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args["learning_rate"]))
    es2_callback = keras.callbacks.EarlyStopping(monitor='cw_reconstruction_loss', patience=args["patience"],
                                                 mode="min")
    model.fit(mnist_digits, epochs=args["epochs"], batch_size=args["batch_size"], callbacks=[es2_callback])

    model.save("high_model.keras", save_format="keras")
    with open("high_model_summary.txt", "w") as f:
        model.encoder.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n")
        model.decoder.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n")
        model.generator.summary(print_fn=lambda x: f.write(x + '\n'))

    loaded_model = keras.saving.load_model("high_model.keras")


if __name__ == "__main__":
    test_saving()
