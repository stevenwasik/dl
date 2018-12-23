import numpy as np
import dl_prop_functions as pf
import dl_helper_functions as hf


def ffnn(target, predictors, classes, hidden_layer_nodes, hidden_layers, predictor_count, eps=0.001, epochs=100):
    b_in = np.reshape(np.random.randn(hidden_layer_nodes), (1, hidden_layer_nodes))
    w_in = np.reshape(np.random.randn(predictor_count * hidden_layer_nodes), (predictor_count, hidden_layer_nodes))

    # if hidden_layers > 1:
    b_hidden = np.reshape(np.random.randn(hidden_layer_nodes), (1, hidden_layer_nodes))
    w_hidden = np.reshape(np.random.randn(hidden_layer_nodes * hidden_layer_nodes),
                          (hidden_layer_nodes, hidden_layer_nodes))

    b_out = np.reshape(np.random.randn(classes), (1, classes))
    w_out = np.reshape(np.random.randn(hidden_layer_nodes * classes), (hidden_layer_nodes, classes))

    for i in range(epochs):
        # Feed Forward Step
        # in layer
        z_in = pf.feed_forward_z(predictors, w_in, b_in)
        a_in = hf.sigmoid(z_in)

        # hidden layer
        # if hidden_layers > 1:
        z_hidden = pf.feed_forward_z(a_in, w_hidden, b_hidden)
        a_hidden = hf.sigmoid(z_hidden)

        # out layer
        z_out = pf.feed_forward_z(a_hidden, w_out, b_out)
        a_out = hf.softmax(z_out)
        cost = hf.cost_function_quadratic(target, a_out)

    # Back Prop Step
    # out layer
    del_out = 2 * np.array(target - a_out) * hf.softmax_delta(z_out)
    b_out = b_out + eps * np.sum(del_out, keepdims=True, axis=0) / predictors.shape[0]

    del_L = np.zeros((hidden_layer_nodes, classes))
    for i in range(classes):
        for j in range(hidden_layer_nodes):
            for k in range(2000):
                del_L[j, i] = 2 * (target[i, k] - a_out[i, k]) * hf.softmax_delta(z[i, :])

    w_out[0, :] = w_out[0, :] + eps * np.sum(np.reshape(np.array(a_hidden[:, 0]), (2000, 1)) * del_out,
                                             keepdims=True,
                                             axis=0) / predictors.shape[0]
    w_out[1, :] = w_out[1, :] + eps * np.sum(np.reshape(np.array(a_hidden[:, 1]), (2000, 1)) * del_out,
                                             keepdims=True,
                                             axis=0) / predictors.shape[0]
    w_out[2, :] = w_out[2, :] + eps * np.sum(np.reshape(np.array(a_hidden[:, 2]), (2000, 1)) * del_out,
                                             keepdims=True,
                                             axis=0) / predictors.shape[0]

    # hidden layers
    # del_hidden =

    # dC / dw2 = dC / dA * dA / dZ * dZ / dX[dZ]

    b_hidden = b_hidden + eps * pf.back_prop_bias(target, z_out, a_out)
    w_hidden[0, :] = w_hidden[0, :] + eps * pf.back_prop_weights_h(target, z_hidden, a_hidden, a_in[:, 0])
    w_hidden[1, :] = w_hidden[1, :] + eps * pf.back_prop_weights_h(target, z_hidden, a_hidden, a_in[:, 1])
    w_hidden[2, :] = w_hidden[2, :] + eps * pf.back_prop_weights_h(target, z_hidden, a_hidden, a_in[:, 2])

    # in layer
    b_in = b_in + eps * pf.back_prop_bias(target, z_out, a_out)
    w_in[0, :] = w_in[0, :] + eps * pf.back_prop_weights_h(target, z_in, a_in, predictors[:, 0])
    w_in[1, :] = w_in[1, :] + eps * pf.back_prop_weights_h(target, z_in, a_in, predictors[:, 1])

    if i % 1000 == 0:
        print('iteration ', i, ' cost: ', cost)

    return a_in, a_hidden, a_out, w_in, w_hidden, w_out, b_in, b_hidden, b_out
