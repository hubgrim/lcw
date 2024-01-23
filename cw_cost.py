import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
import math


def silverman_rule_of_thumb_normal(N: int) -> float:
    return (4 / (3 * N)) ** 0.4


def pairwise_distances(x, y=None):
    if y is None:
        y = x
    x_np = x.numpy()
    y_np = y.numpy()
    distances_np = cdist(x_np, y_np) ** 2
    distances_tf = tf.constant(distances_np)
    return distances_tf


def euclidean_norm_squared(X, axis):
    return tf.math.reduce_sum(tf.math.square(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(X), axis=axis))), axis=-1)


def cw_normality(X, y=None):
    assert len(X.shape) == 2

    N, D = X.shape

    if y is None:
        y = silverman_rule_of_thumb_normal(N)

    K1 = 1.0 / (2.0 * D - 3.0)

    A1 = pairwise_distances(X)
    A = tf.reduce_mean(1 / tf.math.sqrt(y + K1 * A1))

    B1 = euclidean_norm_squared(X, axis=1)
    B = 2 * tf.reduce_mean((1 / tf.math.sqrt(y + 0.5 + K1 * B1)))

    return (1 / math.sqrt(1 + y)) + tf.cast(A, dtype=tf.float32) - B


def phi_sampling(s, D):
    return (1.0 + 4.0 * s / (2.0 * D - 3)) ** (-0.5)


def cw_sampling(first_sample, second_sample, y):
    shape = first_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    first_sample = tf.reshape(first_sample, [-1, dim])

    shape = second_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    second_sample = tf.reshape(second_sample, [-1, dim])

    assert len(first_sample.shape) == 2
    assert first_sample.shape == second_sample.shape

    _, D = first_sample.shape

    T = 1.0 / (2.0 * math.sqrt(math.pi * y))

    A0 = pairwise_distances(first_sample)
    A = tf.reduce_mean(phi_sampling(A0 / (4 * y), D))

    B0 = pairwise_distances(second_sample)
    B = tf.reduce_mean(phi_sampling(B0 / (4 * y), D))

    C0 = pairwise_distances(first_sample, second_sample)
    C = tf.reduce_mean(phi_sampling(C0 / (4 * y), D))

    return T * (A + B - 2 * C)


def cw_sampling_silverman(first_sample, second_sample):
    stddev = tf.math.reduce_std(second_sample)
    N = tf.shape(second_sample)[0]
    gamma = silverman_rule_of_thumb_normal(N)
    return cw_sampling(first_sample, second_sample, gamma)


def cw_cost_function(x, y, z, lambda_val):
    return tf.math.log(cw_sampling_silverman(x, y)) + lambda_val * tf.math.log(cw_normality(z))
