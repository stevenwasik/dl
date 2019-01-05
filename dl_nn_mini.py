import numpy as np
import dl_helper_functions as hf


def nn_mini(target, predictors, classes, hidden_layer_nodes, hidden_layers, predictor_count, eps=0.001, epochs=100,
            minibatch_size=50):
    network_struc = [predictor_count] + [hidden_layer_nodes] * hidden_layers + [classes]
    num_obs = len(target)
    cost = 0

    # Initialize
    layer_dat = [[0, 0, 0, predictors]]
    for i in range(1, len(network_struc)):
        node_count_prior_layer = network_struc[i - 1]
        node_count_current_layer = network_struc[i]
        b = np.reshape(np.random.randn(node_count_current_layer), (1, node_count_current_layer))
        w = np.reshape(np.random.randn(node_count_prior_layer * node_count_current_layer),
                       (node_count_prior_layer, node_count_current_layer))
        z = np.zeros((num_obs, node_count_current_layer))
        a = np.zeros((num_obs, node_count_current_layer))
        layer_dat.append([b, w, z, a])

    for i in range(epochs):
        # Forward Step
        for m in range(0, num_obs, minibatch_size):
            if m + minibatch_size > num_obs:
                minibatch_size = num_obs - m

            for k in range(1, len(layer_dat)):
                layer_dat[k][2][m:m + minibatch_size, :] = np.matmul(layer_dat[k - 1][3][m:m + minibatch_size, :],
                                                                     layer_dat[k][1]) + layer_dat[k][0]
                if k < len(layer_dat):
                    layer_dat[k][3][m:m + minibatch_size, :] = hf.sigmoid(layer_dat[k][2][m:m + minibatch_size, :])
                else:
                    layer_dat[k][3][m:m + minibatch_size, :] = hf.softmax(layer_dat[k][2][m:m + minibatch_size, :])

            # Back Step
            mini_target = target[m:m + minibatch_size, :]
            mini_layer = [layer_dat[-1][0], layer_dat[-1][1],
                          layer_dat[-1][2][m:m + minibatch_size, :], layer_dat[-1][3][m:m + minibatch_size, :]]
            mini_layer_p_a = layer_dat[-2][3][m:m + minibatch_size, :]

            cost = np.mean((mini_target - mini_layer[3]) ** 2)

            del_l = -(mini_target - mini_layer[3]) * hf.softmax_del(mini_layer[2], mini_target)
            del_l_b = np.sum(del_l, keepdims=True, axis=0) / minibatch_size
            del_l_w = np.matmul(np.transpose(mini_layer_p_a), del_l) / minibatch_size
            layer_dat[-1][0] = mini_layer[0] - eps * del_l_b
            layer_dat[-1][1] = mini_layer[1] - eps * del_l_w

            for j in range(2, len(layer_dat)):
                mini_layer = [layer_dat[-j][0], layer_dat[-j][1],
                              layer_dat[-j][2][m:m + minibatch_size, :], layer_dat[-j][3][m:m + minibatch_size, :]]
                mini_layer_p_a = layer_dat[-j - 1][3][m:m + minibatch_size, :]
                mini_layer_n_w = layer_dat[-j + 1][1]

                del_l = np.matmul(del_l, np.transpose(mini_layer_n_w)) * hf.sigmoid_del(mini_layer[2])
                del_l_b = np.sum(del_l, keepdims=True, axis=0) / minibatch_size
                del_l_w = np.matmul(np.transpose(mini_layer_p_a), del_l) / minibatch_size
                layer_dat[-j][0] = mini_layer[0] - eps * del_l_b
                layer_dat[-j][1] = mini_layer[1] - eps * del_l_w

        if i % int(epochs / 10) == 0:
            print('iteration ', i, ' cost: ', cost)

    return cost, layer_dat
