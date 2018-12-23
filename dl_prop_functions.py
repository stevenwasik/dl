import numpy as np
import dl_helper_functions as hf


def feed_forward_z(x, weights, bias):
    return np.matmul(x, weights) + bias


def back_prop_bias(y, z, pred):
    return np.transpose(np.sum(2 * np.array(y - pred) * hf.softmax_delta(z), keepdims=True, axis=0) / len(z))


def back_prop_weights(y, z, pred, x):
    return np.sum(x * 2 * np.array(y - pred) * hf.softmax_delta(z), keepdims=True, axis=0) / len(z)
