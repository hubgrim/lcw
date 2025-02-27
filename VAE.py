import keras
import numpy as np
import tensorflow as tf
from cw_cost import cw_sampling_silverman, cw_normality, mean_squared_euclidean_norm_reconstruction_error, cw_sampling
from architectures import Cw2Encoder, Cw2Decoder, LcwGenerator

class LCW(keras.Model):
    def __init__(self, encoder, decoder, args, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.generator = LcwGenerator(args).build()
        self.args = args
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="cw_reconstruction_loss"
        )
    def get_config(self):
        config = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "args": self.args
        }
        base_config = super(LCW, self).get_config()
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


class CW2(keras.Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Cw2Encoder(args).build()
        self.decoder = Cw2Decoder(args).build()
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
        base_config = super(CW2, self).get_config()
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


class CWAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="cw_reconstruction_loss"
        )
        self.cw_loss_tracker = keras.metrics.Mean(name="cw_loss")

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
            cw_reconstruction_loss = tf.math.log(
                mean_squared_euclidean_norm_reconstruction_error(data, reconstruction))
            lambda_val = 1
            cw_loss = lambda_val * tf.math.log(cw_sampling(z))
            total_loss = tf.cast(cw_reconstruction_loss,
                                 dtype=tf.float32) + cw_loss
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
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
