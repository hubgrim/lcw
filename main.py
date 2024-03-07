from architectures import get_architecture, latent_generator
from VAE import CW2, CWAE, VAE, LCW
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
args["model_path"] = "results/cw2/2024_03_07__13_28_18/model.weights.h5"

args["sample_amount"] = 70000
args["latent_dim"] = 24
args["noise_dim"] = 24
args["epochs"] = 60
args["batch_size"] = 128
args["patience"] = 3
args["learning_rate"] = 0.0001
args["results_dir"] = f"results/"
args["model_type"] = "lcw"
args["architecture_type"] = "lcw"
args["bias"] = True
args["batch_norm"] = True
args["tsne_amount"] = 150

# -------END PARAMETERS-------
(x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)[0:args["sample_amount"]]
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255


if args["model_type"] == "lcw":
    encoder, decoder = get_architecture(args, "lcw")
    cw2_model = CW2(encoder, decoder)
    cw2_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args["learning_rate"]))
    es_callback = keras.callbacks.EarlyStopping(monitor='total_loss', patience=args["patience"], mode="min")
    ts_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    cw2_model.fit(mnist_digits, epochs=args["epochs"], batch_size=args["batch_size"], callbacks=[es_callback])

    generator = latent_generator(args)
    model = LCW(cw2_model.encoder, cw2_model.decoder, generator, args)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args["learning_rate"]))
    es2_callback = keras.callbacks.EarlyStopping(monitor='cw_reconstruction_loss', patience=args["patience"], mode="min")
    ts2_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    model.fit(mnist_digits, epochs=args["epochs"], batch_size=args["batch_size"], callbacks=[es2_callback])

else:
    encoder, decoder = get_architecture(args, args["architecture_type"])

    model = CWAE(encoder, decoder) if args["model_type"] == "cwae" else CW2(
        encoder, decoder) if args["model_type"] == "cw2" else VAE(encoder, decoder)

    if args["load_model"]:
        model.load_weights(args["model_path"])
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=args["learning_rate"]))
        es_callback = keras.callbacks.EarlyStopping(monitor='total_loss', patience=args["patience"], mode="min")
        ts_callback = keras.callbacks.TensorBoard(log_dir="./logs")
        model.fit(mnist_digits, epochs=args["epochs"], batch_size=args["batch_size"], callbacks=[es_callback])

log_results(model, args, x_train, y_train)
