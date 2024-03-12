import tensorflow as tf
import keras
from keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def standard_encoder(args):
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2,
                      padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(args["latent_dim"], name="z_mean")(x)
    z_log_var = layers.Dense(args["latent_dim"], name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(
        encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def standard_decoder(args):
    latent_inputs = keras.Input(shape=(args["latent_dim"],))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(
        64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(
        32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


def lcw_encoder(args):
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(256, use_bias=args["bias"])(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256, use_bias=args["bias"])(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256, use_bias=args["bias"])(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    # z_mean = layers.Dense(args["latent_dim"], name="z_mean")(x)
    # z_log_var = layers.Dense(args["latent_dim"], name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    z = layers.Dense(args["latent_dim"], name="z")(x)
    encoder = keras.Model(encoder_inputs, [z], name="encoder")
    return encoder


@tf.keras.saving.register_keras_serializable()
class Cw2Encoder(keras.Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.activation = layers.Activation("relu")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, use_bias=args["bias"])
        self.batchnorm1 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense2 = layers.Dense(256, use_bias=args["bias"])
        self.batchnorm2 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense3 = layers.Dense(256, use_bias=args["bias"])
        self.batchnorm3 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense4 = layers.Dense(args["latent_dim"], name="z")

    def build(self, **kwargs):
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = self.flatten(encoder_inputs)
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        z = self.dense4(x)
        encoder = keras.Model(encoder_inputs, [z], name="encoder")
        return encoder


@tf.keras.saving.register_keras_serializable()
class Cw2Decoder(keras.Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = args["latent_dim"]
        self.activation = layers.Activation("relu")
        self.dense1 = layers.Dense(256, use_bias=args["bias"])
        self.batchnorm1 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense2 = layers.Dense(256, use_bias=args["bias"])
        self.batchnorm2 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense3 = layers.Dense(256, use_bias=args["bias"])
        self.batchnorm3 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense4 = layers.Dense(28 * 28, activation="sigmoid")
        self.reshape = layers.Reshape([28, 28, 1])

    def build(self, **kwargs):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = self.dense1(latent_inputs)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.dense4(x)
        decoder_outputs = self.reshape(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="encoder")
        return decoder


@tf.keras.saving.register_keras_serializable()
class LcwGenerator(keras.Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.noise_dim = args["noise_dim"]
        self.activation = layers.Activation("relu")
        self.dense1 = layers.Dense(512)
        self.batchnorm1 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense2 = layers.Dense(512)
        self.batchnorm2 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense3 = layers.Dense(512)
        self.batchnorm3 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense4 = layers.Dense(512)
        self.batchnorm4 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense5 = layers.Dense(512)
        self.batchnorm5 = layers.BatchNormalization() if args["batch_norm"] else layers.Identity()
        self.dense6 = layers.Dense(args["latent_dim"], name="z")

    def build(self, **kwargs):
        noise_inputs = keras.Input(shape=(self.noise_dim,))
        x = self.dense1(noise_inputs)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.dense4(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = self.dense5(x)
        x = self.batchnorm5(x)
        x = self.activation(x)
        z = self.dense6(x)
        latent_generator = keras.Model(noise_inputs, [z], name="generator")
        return latent_generator

def lcw_decoder(args):
    latent_inputs = keras.Input(shape=(args["latent_dim"],))
    x = layers.Dense(256, use_bias=args["bias"])(latent_inputs)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256, use_bias=args["bias"])(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256, use_bias=args["bias"])(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    decoder_outputs = layers.Dense(28 * 28, activation="sigmoid")(x)
    decoder_outputs = layers.Reshape([28, 28, 1])(decoder_outputs)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

def latent_generator(args):
    noise_inputs = keras.Input(shape=(args["noise_dim"],))
    x = layers.Dense(512)(noise_inputs)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    z = layers.Dense(args["latent_dim"], name="z")(x)
    latent_generator = keras.Model(noise_inputs, [z], name="generator")
    return latent_generator


def get_architecture(args, architecture_type):
    if architecture_type == "standard":
        return standard_encoder(args), standard_decoder(args)
    elif architecture_type == "lcw":
        return lcw_encoder(args), lcw_decoder(args)
