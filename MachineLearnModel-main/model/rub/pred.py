#!/export/home/tianhaotan/.conda/envs/scikit_0_24andsqlwrite_read_env/bin/python
#KRR

import numpy as np
import math as ma
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.kernel_ridge import KernelRidge
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model._ridge import _solve_cholesky_kernel
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
#from sklearn.ensemble import AdaBoostRegressor
#import pickle 
import joblib
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
with sklearn.config_context(working_memory=8192):
    pass
import torch
import torch.nn as nn
from torch.autograd import Variable as var
import torch.utils.data as Data
torch.set_default_tensor_type(torch.DoubleTensor)

m = 10
cm =open('A321DIY.txt','r')
cm = cm.read()
cm = cm.split()
cm = np.array(cm).reshape((-1,3200))
cm = cm.astype(float)


x_mean =open('edit1mean','r')
x_mean = x_mean.read()
x_mean = x_mean.split()
x_mean = np.array(x_mean).reshape(-1,3200)
x_mean = x_mean.astype(float)
x_std =open('edit1std','r')
x_std = x_std.read()
x_std = x_std.split()
x_std = np.array(x_std).reshape(-1,3200)
x_std = x_std.astype(float)

class MultipleLayerRegressor(nn.Module):
   def __init__(self, n_feature, n_hid1, n_hid2, n_output):
       super(MultipleLayerRegressor, self).__init__()
       self.hidden = torch.nn.Linear(n_feature, n_hid1)
       self.inner1 = torch.nn.Linear(n_hid1, n_hid1)
       self.inner2 = torch.nn.Linear(n_hid1, n_hid1)
       self.inner3 = torch.nn.Linear(n_hid1, n_hid2)
       self.inner4 = torch.nn.Linear(n_hid2, n_hid2)
       self.inner5 = torch.nn.Linear(n_hid2, n_hid2)
       self.inner6 = torch.nn.Linear(n_hid2, n_hid2)
       self.out = torch.nn.Linear(n_hid2, n_output)
   def forward(self, x):
       x = torch.relu(self.hidden(x))      # activation function for hidden layer
       x = torch.relu(self.inner1(x))
       x = torch.relu(self.inner2(x))
       x = torch.relu(self.inner3(x))
       x = torch.relu(self.inner4(x))
       x = torch.relu(self.inner5(x))
       x = torch.relu(self.inner6(x))
       x = self.out(x)
       return x

mlp=torch.load('mlp.pth')
np.savetxt('predDIY',mlp(var(torch.from_numpy(x))).detach().numpy(), fmt='%s',delimiter=' ' )


