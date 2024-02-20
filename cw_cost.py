import numpy as np
import keras
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.spatial.distance import cdist
import math


def silverman_rule_of_thumb_normal(N):
    return tf.pow((4 / (3 * N)), 0.4)


def pairwise_distances(x, y=None):
    if y is None:
        y = x
    distances_tf = tf.norm(x[:, None] - y, axis=-1) ** 2
    return tf.cast(distances_tf, dtype=tf.float64)


def cw_normality(X, y=None):
    assert len(X.shape) == 2

    D = tf.cast(tf.shape(X)[1], tf.float64)
    N = tf.cast(tf.shape(X)[0], tf.float64)

    if y is None:
        y = silverman_rule_of_thumb_normal(N)

    # adjusts for dimensionality; D=2 -> K1=1, D>2 -> K1<1
    K1 = 1.0 / (2.0 * D - 3.0)

    A1 = pairwise_distances(X)
    A = tf.reduce_mean(1 / tf.math.sqrt(y + K1 * A1))

    B1 = tf.cast(tf.square(tf.math.reduce_euclidean_norm(X, axis=1)), dtype=tf.float64)
    B = 2 * tf.reduce_mean((1 / tf.math.sqrt(y + 0.5 + K1 * B1)))

    return (1 / tf.sqrt(1 + y)) + A - B


def phi_sampling(s, D):
    return tf.pow(1.0 + 4.0 * s / (2.0 * D - 3), -0.5)


def cw_sampling_lcw(first_sample, second_sample, y):
    shape = first_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    first_sample = tf.reshape(first_sample, [-1, dim])

    shape = second_sample.get_shape().as_list()
    dim = np.prod(shape[1:])
    second_sample = tf.reshape(second_sample, [-1, dim])

    assert len(first_sample.shape) == 2
    assert first_sample.shape == second_sample.shape

    _, D = first_sample.shape

    T = 1.0 / (2.0 * tf.sqrt(math.pi * y))

    A0 = pairwise_distances(first_sample)
    A = tf.reduce_mean(phi_sampling(A0 / (4 * y), D))

    B0 = pairwise_distances(second_sample)
    B = tf.reduce_mean(phi_sampling(B0 / (4 * y), D))

    C0 = pairwise_distances(first_sample, second_sample)
    C = tf.reduce_mean(phi_sampling(C0 / (4 * y), D))

    return T * (A + B - 2 * C)


def euclidean_norm_squared(X, axis=None):
    return tf.reduce_sum(tf.square(X), axis=axis)


def squared_euclidean_norm_reconstruction_error(input, output):
    return euclidean_norm_squared(input - output, axis=1)


def mean_squared_euclidean_norm_reconstruction_error(x, y):
    return tf.reduce_mean(
        squared_euclidean_norm_reconstruction_error(keras.layers.Flatten()(x), keras.layers.Flatten()(y)))


def cw_sampling(X, y=None):
    def phi_sampling(s, D):
        return tf.pow(1.0 + 4.0 * s / (2.0 * D - 3), -0.5)

    D = tf.cast(tf.shape(X)[1], tf.float32)
    N = tf.cast(tf.shape(X)[0], tf.float32)
    D_int = tf.cast(D, tf.int32)
    N_int = tf.cast(N, tf.int32)
    if y is None:
        y = silverman_rule_of_thumb_normal(N)

    YDistr = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(D_int, tf.float32),
                                                      scale_diag=tf.ones(D_int, tf.float32))
    Y = YDistr.sample(N_int)
    T = 1.0 / (2.0 * N * tf.sqrt(math.pi * y))

    A0 = euclidean_norm_squared(tf.subtract(
        tf.expand_dims(X, 0), tf.expand_dims(X, 1)), axis=2)
    A = tf.reduce_sum(phi_sampling(A0 / (4 * y), D))

    B0 = euclidean_norm_squared(tf.subtract(
        tf.expand_dims(Y, 0), tf.expand_dims(Y, 1)), axis=2)
    B = tf.reduce_sum(phi_sampling(B0 / (4 * y), D))

    C0 = euclidean_norm_squared(tf.subtract(
        tf.expand_dims(X, 0), tf.expand_dims(Y, 1)), axis=2)
    C = tf.reduce_sum(phi_sampling(C0 / (4 * y), D))

    return T * (A + B - 2 * C)


def cw_sampling_silverman(first_sample, second_sample):
    stddev = tf.math.reduce_std(second_sample)
    N = tf.cast(tf.shape(second_sample)[0], tf.float64)
    gamma = silverman_rule_of_thumb_normal(N)
    return cw_sampling_lcw(first_sample, second_sample, gamma)


def cw_cost_function(x, y, z, lambda_val):
    return tf.math.log(cw_sampling_silverman(x, y)) + lambda_val * tf.math.log(cw_normality(z))
