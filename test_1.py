# Implementation of Feed Forward Neural Network Classifer using gradient descent

import numpy as np
import pandas as pd
import dl_ffnn

c0_mean = [2, -2]
c0_cov = [[1, 0], [0, 1]]
c0 = np.random.multivariate_normal(c0_mean, c0_cov, 500)
c0_df = pd.DataFrame(c0)
c0_df.columns = ['x', 'y']
c0_df['group'] = 0.0
c0_df['t0'] = 1.0
c0_df['t1'] = 0.0
c0_df['t2'] = 0.0
c0_df['t3'] = 0.0

c1_mean = [-2, -2]
c1_cov = [[1, 0], [0, 1]]
c1 = np.random.multivariate_normal(c1_mean, c1_cov, 500)
c1_df = pd.DataFrame(c1)
c1_df.columns = ['x', 'y']
c1_df['group'] = 1.0
c1_df['t0'] = 0.0
c1_df['t1'] = 1.0
c1_df['t2'] = 0.0
c1_df['t3'] = 0.0

c2_mean = [2, 2]
c2_cov = [[1, 0], [0, 1]]
c2 = np.random.multivariate_normal(c2_mean, c2_cov, 500)
c2_df = pd.DataFrame(c2)
c2_df.columns = ['x', 'y']
c2_df['group'] = 1.0
c2_df['t0'] = 0.0
c2_df['t1'] = 1.0
c2_df['t2'] = 0.0
c2_df['t3'] = 0.0

c3_mean = [-2, 2]
c3_cov = [[1, 0], [0, 1]]
c3 = np.random.multivariate_normal(c3_mean, c3_cov, 500)
c3_df = pd.DataFrame(c3)
c3_df.columns = ['x', 'y']
c3_df['group'] = 0.0
c3_df['t0'] = 1.0
c3_df['t1'] = 0.0
c3_df['t2'] = 0.0
c3_df['t3'] = 0.0

dat = pd.concat([c0_df, c1_df, c2_df, c3_df], ignore_index=True)

# dat.plot.scatter(x='x', y='y', c='group', cmap='coolwarm')

target = np.array(dat[['t0', 't1']])
predictors = np.array(dat[['x', 'y']])

hidden_layers = 2
hidden_layer_nodes = 3
classes = 2
predictor_count = 2

# a_in, a_hidden, a_out, w_in, w_hidden, w_out, b_in, b_hidden, b_out = dl_ffnn.ffnn(target, predictors, classes,
#                                                                                   hidden_layer_nodes, hidden_layers,
#                                                                                   predictor_count, eps=0.001,
#                                                                                   epochs=100)

# dat['pred_0'] = np.round(a_out[:, 0], 0)

# dat['pred_desc'] = dat[['t0', 'pred_0']].apply(lambda df: 1 if df['t0'] == df['pred_0'] else 0, axis=1)

# dat.groupby(['t0', 'pred_0'])['pred_desc'].count()

# dat.plot.scatter(x='x', y='y', c='pred_0', cmap='coolwarm')
