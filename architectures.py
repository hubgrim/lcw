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
    x = layers.Dense(256)(noise_inputs)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x) if args["batch_norm"] else x
    x = layers.Activation("relu")(x)
    z = layers.Dense(args["latent_dim"], name="z")(x)
    latent_generator = keras.Model(noise_inputs, [z], name="latent_generator")
    return latent_generator

def get_architecture(args):
    if args["architecture_type"] == "standard":
        return standard_encoder(args), standard_decoder(args)
    elif args["architecture_type"] == "lcw":
        return lcw_encoder(args), lcw_decoder(args)
