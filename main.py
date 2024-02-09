import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from datetime import datetime
from plotting import plot_latent_space, plot_label_clusters
from VAE import CWAE, VAE
from architecures import standard_encoder, standard_decoder

tf.config.run_functions_eagerly(True)

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

latent_dim = 8
encoder = standard_encoder(latent_dim)
decoder = standard_decoder(latent_dim)

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
ts_callback = keras.callbacks.TensorBoard(log_dir="./logs")
vae.fit(mnist_digits, epochs=1, batch_size=128, callbacks=[es_callback, ts_callback])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'_{timestamp}.png'

plot_latent_space(vae, filename)

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train, filename)
