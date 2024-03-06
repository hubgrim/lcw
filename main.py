from architectures import get_architecture
from VAE import CW2, CWAE, VAE
from utils import log_results
import keras
import tensorflow as tf
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"


# tf.config.run_functions_eagerly(True)

# -------BEGIN PARAMETERS-------
args = {}
args["load_model"] = False
args["model_path"] = "results/cw2/lat_dim2_epochs_1_20240301_151654/model.weights.h5"

args["latent_dim"] = 24
args["epochs"] = 1
args["batch_size"] = 128
args["patience"] = 3
args["results_dir"] = f"results/"
args["model_type"] = "cw2"
args["architecture_type"] = "lcw"
args["bias"] = False
args["batch_norm"] = True
args["tsne_amount"] = 150

# -------END PARAMETERS-------
(x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()

encoder, decoder = get_architecture(args)

model = CWAE(encoder, decoder) if args["model_type"] == "cwae" else CW2(
    encoder, decoder) if args["model_type"] == "cw2" else VAE(encoder, decoder)

if args["load_model"]:
    model.load_weights(args["model_path"])
else:
    mnist_digits = np.concatenate([x_train, x_test], axis=0)[0:1000]
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    es_callback = keras.callbacks.EarlyStopping(monitor='total_loss', patience=args["patience"], mode="min")
    ts_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    model.fit(mnist_digits, epochs=args["epochs"], batch_size=args["batch_size"], callbacks=[es_callback])

log_results(model, args, x_train, y_train)
