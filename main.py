from architectures import get_architecture, latent_generator
from VAE import CW2, CWAE, VAE, LCW
from utils import log_results
import keras
import tensorflow as tf
import numpy as np
import os
import h5py

os.environ["KERAS_BACKEND"] = "tensorflow"


# tf.config.run_functions_eagerly(True)

# -------BEGIN PARAMETERS-------
args = {"load_model": True,
        "model_path": "results/lcw/2024_03_12__11_56_49/model.keras",
        "sample_amount": 1000,
        "latent_dim": 24,
        "noise_dim": 24,
        "epochs": 1,
        "batch_size": 128,
        "patience": 3,
        "learning_rate": 0.0001,
        "results_dir": f"results/",
        "model_type": "lcw",
        "architecture_type": "lcw",
        "bias": True,
        "batch_norm": True,
        "tsne_amount": 500,
        "perplexity": 10}

# -------END PARAMETERS-------
(x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)[0:args["sample_amount"]]
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255


if args["model_type"] == "lcw":
    encoder, decoder = get_architecture(args, "lcw")
    generator = latent_generator(args)
    if args["load_model"]:
        model = keras.saving.load_model(args["model_path"])
    else:
        cw2_model = CW2(args)
        cw2_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args["learning_rate"]))
        es_callback = keras.callbacks.EarlyStopping(monitor='total_loss', patience=args["patience"], mode="min")
        ts_callback = keras.callbacks.TensorBoard(log_dir="./logs")
        cw2_model.fit(mnist_digits, epochs=args["epochs"], batch_size=args["batch_size"], callbacks=[es_callback])

        model = LCW(cw2_model.encoder, cw2_model.decoder, args)
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
