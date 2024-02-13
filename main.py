import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from datetime import datetime
from plotting import plot_latent_space, plot_label_clusters
from VAE import CWAE, VAE
from architecures import standard_encoder, standard_decoder, lcw_encoder, lcw_decoder

tf.config.run_functions_eagerly(True)

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

latent_dim = 2
epochs=60
encoder = lcw_encoder(latent_dim)
decoder = lcw_decoder(latent_dim)

vae = CWAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
ts_callback = keras.callbacks.TensorBoard(log_dir="./logs")
vae.fit(mnist_digits, epochs=epochs, batch_size=128)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_type = "VAE" if isinstance(vae, VAE) else "CWAE"
filename = f'{model_type}_lat_dim{latent_dim}_epochs_{epochs}'
plots_dir = f"results/plots/{timestamp}/"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

plot_latent_space(vae, plots_dir+filename+"_latent_space.png")

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train, plots_dir+filename+"_label_clusters.png")
