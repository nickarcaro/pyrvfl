import numpy as np


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def sine(x):
    return np.sin(x)


def hardlim(x):
    return (np.sign(x) + 1) / 2


def tribas(x):
    return np.maximum(1 - np.abs(x), 0)


def radbas(x):
    return np.exp(-(x**2))


def sign(x):
    return np.sign(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)


def softmax(x):
    return np.exp(x) / np.repeat(
        (np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1
    )
