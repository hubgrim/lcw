import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from datetime import datetime
from sklearn.manifold import TSNE


def plot_latent_space(model, data, labels, saving_path, perplexity):
    # copied from amp-cevae/gmmvae/amp-vae/scripts/clustering.py and modified
    print("-- Start t-SNE plot --")
    tsne_model = TSNE(n_components=2,
                      random_state=42,
                      n_jobs=-1,
                      verbose=1,
                      perplexity=perplexity,
                      init='pca',
                      )
    z = model.encoder.predict(data, verbose=0)
    z = tsne_model.fit_transform(z)
    z_0 = []
    z_1 = []

    for i in z:
        z_0.append(i[0])
        z_1.append(i[1])

    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(z_0, z_1, c=labels, s=30, cmap='viridis')
    # Get unique labels
    unique_labels = np.unique(labels)

    # Create legend entries with specific labels and colors
    legend_entries = [
        plt.Line2D([0], [0], marker='o', color=scatter.cmap(scatter.norm(label)), markersize=10, label=f'{label}')
        for label in unique_labels]

    # Add legend
    plt.legend(handles=legend_entries, title='Legend', loc='upper right')

    # create DataFrame of z_0 = 'z_0', z_1 = 'z_1', label = 'label'
    df = pd.DataFrame({"z_0": z_0, "z_1": z_1, "labels": labels})
    # sns.kdeplot(df, x="z_0", y="z_1", hue="labels", fill=True, thresh=0.2,
    #             alpha=0.75, palette=[
    #         "#FF5733",  # Vivid Orange
    #         "#3498DB",  # Bright Blue
    #         "#FFD700",  # Gold
    #         "#8E44AD",  # Rich Purple
    #         "#2ECC71",  # Emerald Green
    #         "#E74C3C",  # Fiery Red
    #         "#1ABC9C",  # Turquoise
    #         "#F39C12",  # Vibrant Yellow
    #         "#9B59B6",  # Royal Purple
    #         "#27AE60"  # Fresh Green
    #     ]
    #             )

    plt.title(f"Latent-Space")
    plt.xlabel("T1")
    plt.ylabel("T2")
    # plt.legend([f"MIC < {lower_threshold}", f"MIC > {upper_threshold}"])
    plt.savefig(saving_path)


def plot_latent_space_samples(model, latent_dim, filename, n=30, figsize=15):
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


def log_results(model, model_type, latent_dim, epochs, results_dir, x_train, y_train, tsne_amount):
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    save_dir = results_dir + f'{model_type}/{timestamp}/'
    plots_dir = save_dir + "plots/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    model.save_weights(save_dir + "model.weights.h5")

    plot_latent_space(model=model, data=x_train[0:tsne_amount], labels=y_train[0:tsne_amount],
                      saving_path=plots_dir + "tsne.png", perplexity=30)

    plot_latent_space_samples(model, latent_dim, plots_dir + "samples.png")

    if latent_dim == 2:
        x_train = np.expand_dims(x_train, -1).astype("float32") / 255

        plot_label_clusters(model, x_train, y_train, plots_dir + "label_clusters.png")
