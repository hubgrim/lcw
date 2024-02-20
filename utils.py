import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def plot_latent_space(model, latent_dim, filename, n=30, figsize=15):
    # display an n*n 2D manifold of digits in the latent space
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))

    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid = np.linspace(-scale, scale, n)
    if latent_dim == 2:
        grid_x, grid_y = np.meshgrid(grid, grid)
        coordinates = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    else:
        coordinates = np.random.uniform(-scale, scale, size=(n * n, latent_dim))

    for i, z_sample in enumerate(coordinates):
        x_decoded = model.decoder.predict(np.array([z_sample]), verbose=0)
        digit = x_decoded[0].reshape(digit_size, digit_size)

        if latent_dim == 2:
            j, k = divmod(i, n)
        else:
            j, k = divmod(i, n)

        figure[
        j * digit_size: (j + 1) * digit_size,
        k * digit_size: (k + 1) * digit_size,
        ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)

    plt.xticks(pixel_range, [])
    plt.yticks(pixel_range, [])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]" if latent_dim == 2 else "z[1], z[2], ...")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(filename, dpi=300)


def plot_label_clusters(model, data, labels, filename):
    # display a 2D plot of the digit classes in the latent space
    z_mean = model.encoder.predict(data, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename, dpi=300)


def log_results(model, model_type, latent_dim, epochs, results_dir, load_model, x_train, y_train):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = results_dir + f'{model_type}_lat_dim{latent_dim}_epochs_{epochs}_{timestamp}/'
    plots_dir = save_dir + "plots/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # if not load_model:    config lacking info to restore object(decoder encoder)
    #     model.save(save_dir + "model.keras")

    plot_latent_space(model, latent_dim, plots_dir + "latent_space.png")

    if latent_dim == 2:
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255

        plot_label_clusters(model, x_train, y_train, plots_dir + "label_clusters.png")
