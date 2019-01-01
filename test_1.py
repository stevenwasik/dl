# Implementation of Feed Forward Neural Network Classifer using gradient descent

import numpy as np
import pandas as pd
import dl_nn

c0_mean = [2, 2]
c0_cov = [[1, 0], [0, 1]]
c0 = np.random.multivariate_normal(c0_mean, c0_cov, 500)
c0_df = pd.DataFrame(c0)
c0_df.columns = ['x', 'y']
c0_df['group'] = 0.0
c0_df['t0'] = 1.0
c0_df['t1'] = 0.0
c0_df['t2'] = 0.0

c1_mean = [-2, -2]
c1_cov = [[1, 0], [0, 1]]
c1 = np.random.multivariate_normal(c1_mean, c1_cov, 500)
c1_df = pd.DataFrame(c1)
c1_df.columns = ['x', 'y']
c1_df['group'] = 1.0
c1_df['t0'] = 0.0
c1_df['t1'] = 1.0
c1_df['t2'] = 0.0

c2_mean = [2, -2]
c2_cov = [[1, 0], [0, 1]]
c2 = np.random.multivariate_normal(c2_mean, c2_cov, 500)
c2_df = pd.DataFrame(c2)
c2_df.columns = ['x', 'y']
c2_df['group'] = 2.0
c2_df['t0'] = 0.0
c2_df['t1'] = 0.0
c2_df['t2'] = 1.0

c3_mean = [-2, 2]
c3_cov = [[1, 0], [0, 1]]
c3 = np.random.multivariate_normal(c3_mean, c3_cov, 500)
c3_df = pd.DataFrame(c3)
c3_df.columns = ['x', 'y']
c3_df['group'] = 2.0
c3_df['t0'] = 0.0
c3_df['t1'] = 1.0

dat = pd.concat([c0_df, c1_df, c2_df], ignore_index=True)

dat.plot.scatter(x='x', y='y', c='group', cmap='coolwarm')

# print(dat)

target = np.array(dat[['t0', 't1', 't2']])
predictors = np.array(dat[['x', 'y']])

hidden_layers = 2
hidden_layer_nodes = 3
classes = 3
predictor_count = 2

cost, layer_dat = dl_nn.nn(target, predictors, classes, hidden_layer_nodes, hidden_layers, predictor_count, eps=0.005,
                           epochs=10000)

dat['pred_group'] = np.argmax(layer_dat[-1][3], axis=1)

# print(dat.groupby(['t0', 'pred_0'])['pred_desc'].count())

dat.plot.scatter(x='x', y='y', c='pred_group', cmap='coolwarm')
