import numpy as np


@staticmethod
def get_random_vectors(m, n, scale_range):
    x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
    return x


@staticmethod
def one_hot(x, n_class):
    y = np.zeros([len(x), n_class])
    for i in range(len(x)):
        y[i, x[i]] = 1
    return y


@staticmethod
def standardize_rvfl(x):
    data_std = np.maximum(np.std(x, axis=0), 1 / np.sqrt(len(x)))
    data_mean = np.mean(x, axis=0)
    return (x - data_mean) / data_std


@staticmethod
def standardize_deep(x, index):
    data_std[index] = np.maximum(np.std(x, axis=0), 1 / np.sqrt(len(x)))
    data_mean[index] = np.mean(x, axis=0)
    return (x - data_mean[index]) / data_std[index]


@staticmethod
def softmax(x):
    return np.exp(x) / np.repeat(
        (np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1
    )
