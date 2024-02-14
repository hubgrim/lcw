import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from utils import log_results
from VAE import CWAE, VAE
from architectures import get_architecture

tf.config.run_functions_eagerly(True)

# -------BEGIN PARAMETERS-------
load_model = False
model_path = ""

latent_dim = 8
epochs = 60
batch_size = 128
patience = 3
results_dir = f"results/"
model_type = "CWAE"
architecture_type = "lcw"

# -------END PARAMETERS-------
if load_model:
    model = tf.keras.models.load_model(model_path)
else:
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    encoder, decoder = get_architecture(architecture_type, latent_dim)

    model = CWAE(encoder, decoder) if model_type == "CWAE" else VAE(encoder, decoder)
    model.compile(optimizer=keras.optimizers.Adam())
    es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    ts_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    model.fit(mnist_digits, epochs=epochs, batch_size=batch_size)

log_results(model, model_type, latent_dim, epochs, results_dir, load_model)
