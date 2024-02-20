import torch
import tensorflow as tf
import keras
import numpy as np
from scipy.spatial.distance import cdist


def pairwise_distances_tf(x, y=None):
    if y is None:
        y = x
    x_np = x.numpy()
    y_np = y.numpy()

    # euclidean distance squared
    distances_np = cdist(x_np, y_np) ** 2

    distances_tf = tf.constant(distances_np)
    return distances_tf


def pairwise_distances_tf2(x, y=None):
    if y is None:
        y = x
    distances_tf = tf.norm(x[:, None] - y, axis=-1) ** 2
    return distances_tf


def pairwise_distances_torch(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
    if y is None:
        y = x
    return torch.cdist(x, y) ** 2


tf_tensor = tf.constant([[1., 1.,1.], [3., 2.,1.]])
torch_tensor = torch.tensor([[1., 1.,1.], [3., 2.,1.]])
pairwise_distances_tensorflow = pairwise_distances_tf(tf_tensor)
pairwise_distances_pytorch = pairwise_distances_torch(torch_tensor)
pairwise_distances_tensorflow2 = pairwise_distances_tf2(tf_tensor)
torch_n = torch_tensor.size(0)
tf_n = tf.shape(tf_tensor)[0]

print("done")


