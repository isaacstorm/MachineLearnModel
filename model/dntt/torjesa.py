#!/export/home/tianhaotan/.conda/envs/scikit_0_24andsqlwrite_read_env/bin/python

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

print(torch.cuda.is_available())

m = 24000
cm =open('A321exx0_edit1.txt','r')
cm = cm.read()
cm = cm.split()
cm = np.array(cm).reshape((-1,4600))
cm = cm.astype(float)
jeff =open('jefflod0.txt','r')
jeff = jeff.read()
jeff = jeff.split()
jeff = np.array(jeff)
jeff = jeff.astype(float)

x = cm[:m]
y = jeff.reshape(-1,1)
y= y*27.2
y = y[:m]
print(x.shape)
x_mean = x.mean(axis=0)
x_scale = np.std(x, axis=0)
y_mean = y.mean()
y_scale = np.std(y)
print(y_mean,y_scale)
x = (x - x_mean) / x_scale
np.savetxt('edit1mean',x_mean, fmt='%s',delimiter=' ' )
np.savetxt('edit1std',x_scale, fmt='%s',delimiter=' ' )
print('finish')

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
x_train,x_test,y_train,y_test = var(torch.from_numpy(train_x)),var(torch.from_numpy(test_x)),var(torch.from_numpy(train_y)),var(torch.from_numpy(test_y))

trainset =  Data.TensorDataset(x_train, y_train)
testset =  Data.TensorDataset(x_test, y_test)
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

mlp = MultipleLayerRegressor(n_feature=4600,n_hid1=6900,n_hid2=576,n_output=1)
loss_fn = torch.nn.L1Loss(reduction='mean')
learning_rate = 1e-5
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=5E-5) 
bat_siz = 500

train_loader = Data.DataLoader(
    dataset=trainset,      # torch TensorDataset format
    batch_size=bat_siz,      # mini batch size
    shuffle=True,               # random shuffle for training
    )
test_loader = Data.DataLoader(
    dataset=testset,      # torch TensorDataset format
    batch_size=bat_siz,      # mini batch size
    shuffle=True,               # random shuffle for training
    )
def train_loop(dataloader, model, loss_fn, opt):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
#        print(X.shape)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
epoch = 600
for t in range(epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, mlp, loss_fn, optimizer)
    test_loop(test_loader, mlp, loss_fn)
#    print(mlp(x_test).detach().numpy())
    if t % 100 == 0:
        print("Error on training set: %g ev" % (np.abs(mlp(x_train).detach().numpy() - train_y).mean()))
        print("Error on test set: %g ev" % (np.abs(mlp(x_test).detach().numpy() - test_y).mean() ))
        print("R-square on training set: %g" % r2_score(train_y, mlp(x_train).detach().numpy()))
        print("R-square on test set: %g" % r2_score(test_y, mlp(x_test).detach().numpy()))
    print("Error on training set: %g ev" % (np.abs(mlp(x_train).detach().numpy() - train_y).mean()))
    print("Error on test set: %g ev" % (np.abs(mlp(x_test).detach().numpy() - test_y).mean() ))
    print("R-square on training set: %g" % r2_score(train_y, mlp(x_train).detach().numpy()))
    print("R-square on test set: %g" % r2_score(test_y, mlp(x_test).detach().numpy()))
torch.save(mlp, 'mlp.pth')
