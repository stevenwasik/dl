import numpy as np


def cost_function_quadratic(y, pred):
    return np.mean((y - pred) ** 2, axis=0)


def sigmoid(x):
    return (1 + np.exp(-x)) ** -1


def sigmoid_del(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    tot = np.sum(np.exp(x), keepdims=True, axis=1)
    return np.exp(x) / tot


def softmax_del(x, t):
    x = softmax(x)
    t_z = x
    t_z[t == 1] = 1 - t_z[t == 1]
    t_z[t == 0] = -t_z[t == 0]

    return x * t_z
