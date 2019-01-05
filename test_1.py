# Implementation of Feed Forward Neural Network Classifer using gradient descent
from __future__ import print_function, division
import numpy as np
import pandas as pd
import dl_nn
import dl_nn_mini
import time

from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import matplotlib.pyplot as plt



c0_mean = [2, 2]
c0_cov = [[1, 0], [0, 1]]
c0 = np.random.multivariate_normal(c0_mean, c0_cov, 500000)
c0_df = pd.DataFrame(c0)
c0_df.columns = ['x', 'y']
c0_df['group'] = 1.0
c0_df['t0'] = 0.0
c0_df['t1'] = 1.0
c0_df['t2'] = 0.0


c1_mean = [-2, -2]
c1_cov = [[1, 0], [0, 1]]
c1 = np.random.multivariate_normal(c1_mean, c1_cov, 500000)
c1_df = pd.DataFrame(c1)
c1_df.columns = ['x', 'y']
c1_df['group'] = 1.0
c1_df['t0'] = 0.0
c1_df['t1'] = 1.0
c1_df['t2'] = 0.0

c2_mean = [2, -2]
c2_cov = [[1, 0], [0, 1]]
c2 = np.random.multivariate_normal(c2_mean, c2_cov, 500000)
c2_df = pd.DataFrame(c2)
c2_df.columns = ['x', 'y']
c2_df['group'] = 0.0
c2_df['t0'] = 1.0
c2_df['t1'] = 0.0
c2_df['t2'] = 0.0

c3_mean = [-2, 2]
c3_cov = [[1, 0], [0, 1]]
c3 = np.random.multivariate_normal(c3_mean, c3_cov, 500000)
c3_df = pd.DataFrame(c3)
c3_df.columns = ['x', 'y']
c3_df['group'] = 2.0
c3_df['t0'] = 0.0
c3_df['t1'] = 0.0
c3_df['t2'] = 1.0

dat = pd.concat([c0_df, c1_df, c2_df, c3_df], ignore_index=True)

dat.plot.scatter(x='x', y='y', c='group', cmap='coolwarm')


target = np.array(dat[['t0', 't1', 't2']])
predictors = np.array(dat[['x', 'y']])

hidden_layers = 1
hidden_layer_nodes = 4
classes = 3
predictor_count = 2

start = time.time()

cost, layer_dat = dl_nn.nn(target, predictors, classes, hidden_layer_nodes, hidden_layers, predictor_count, eps=0.01,
                           epochs=100)
end = time.time()

dat['pred_group'] = np.argmax(layer_dat[-1][3], axis=1)  # np.round(layer_dat[-1][3],0)

print(dat.groupby(['group', 'pred_group']).size().reset_index())
dat.plot.scatter(x='x', y='y', c='pred_group', cmap='coolwarm')

start2 = time.time()
cost, layer_dat = dl_nn_mini.nn_mini(target, predictors, classes, hidden_layer_nodes, hidden_layers, predictor_count,
                                     eps=0.01,
                                     epochs=100, minibatch_size=10000)

end2 = time.time()
print(end - start)
print(end2 - start2)


dat['pred_group'] = np.argmax(layer_dat[-1][3], axis=1)  # np.round(layer_dat[-1][3],0)

print(dat.groupby(['group', 'pred_group']).size().reset_index())
dat.plot.scatter(x='x', y='y', c='pred_group', cmap='coolwarm')
