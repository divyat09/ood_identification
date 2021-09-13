#Common imports
import sys
import os
import argparse
import random
import copy

import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA


from data.data_loader import BaseDataLoader
from models.fc_regression import FC


def compute_rmse(pred_z, z):
    print(np.std(z))
    z= (z - np.mean(z))/ np.std(z)
    pred_z= (pred_z - np.mean(pred_z))/ np.std(pred_z)
        
    print(np.sqrt( np.mean((z - pred_z)**2)) )
    

# Linear Regression between z and z_hat
def linear_regression_approx(z, z_pred):
    return np.matmul( np.linalg.inv( 1e-8 + np.matmul(z_pred.transpose(), z_pred)), np.matmul(z_pred.transpose(), z)  )

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_dim', type=int, default= 2, 
                    help='')
parser.add_argument('--num_tasks', type=int, default=1, 
                    help='')
parser.add_argument('--batch_size', type=int, default= 16, 
                    help='')
parser.add_argument('--lr', type=float, default= 0.01, 
                    help='')
parser.add_argument('--num_epochs', type=int, default= 20, 
                    help='')
parser.add_argument('--num_seeds', type=int, default=3, 
                    help='')

args = parser.parse_args()
batch_size= args.batch_size
lr= args.lr
num_epochs= args.num_epochs
data_dim= args.data_dim
num_tasks= args.num_tasks
num_seeds= args.num_seeds

# Load Dataset
kwargs={}
data_obj= BaseDataLoader(data_case='train')
train_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

data_obj= BaseDataLoader(data_case='test')
test_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

res={}

for seed in range(1, 1+num_seeds):
    
    #Seed values
    random.seed(seed*10)
    np.random.seed(seed*10) 
    torch.manual_seed(seed*10)
    
    #Load Model
    model= FC(data_dim, num_tasks)

    #Optimizer
    opt= optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, model.parameters()) }, 
                    ], lr= lr, weight_decay= 5e-4, momentum= 0.9,  nesterov=True ) 

    #MSE Loss
    loss_function= nn.MSELoss()
    
    print('')
    #Training
    for epoch in range(num_epochs):
        train_loss=0.0
        count=0
        for batch_idx, (x, y, z) in enumerate(train_dataset):

            # Forward Pass
            out= model(x)
            loss= torch.mean(torch.sum((out-y)**2, dim=1))

            #Backward Pass
            loss.backward()
            opt.step()
            opt.zero_grad()

            train_loss+= loss.item()
            count+=1

#         print('Done Training for Epoch: ', epoch)
#         print('MSE Loss: ', train_loss)


#     #Test
#     model.eval()
#     true_y=[]
#     pred_y=[]
#     for batch_idx, (x, y, z) in enumerate(train_dataset):
#         true_y.append(y)
#         pred_y.append(model(x))

#     true_y= torch.cat(true_y).detach().numpy()
#     pred_y= torch.cat(pred_y).detach().numpy()

#     print(true_y.shape, pred_y.shape)
#     plt.plot(range(true_y.shape[0]), pred_y, label='Predicted Var' )
#     plt.plot(range(true_y.shape[0]), true_y, label='True Var' )
#     plt.legend()
#     plt.savefig('train_res.png')
#     plt.clf()

    #Test
    model.eval()
    true_y=[]
    pred_y=[]
    true_z= []
    pred_z= []
    for batch_idx, (x, y, z) in enumerate(test_dataset):

        true_z.append(z)
        pred_z.append(model.rep_net(x))

        true_y.append(y)
        pred_y.append(model(x))

    true_z= torch.cat(true_z).detach().numpy()
    pred_z= torch.cat(pred_z).detach().numpy()

    true_y= torch.cat(true_y).detach().numpy()
    pred_y= torch.cat(pred_y).detach().numpy()
    
    target_pred_err= np.sqrt(np.mean((true_y - pred_y)**2))
    key= 'target_pred_rmse'
    if key not in res.keys():
        res[key]= []
    res[key].append(target_pred_err)    
    
    for idx in range(true_y.shape[1]):
        plt.plot(range(true_y.shape[0]), pred_y[:, idx], label='Predicted Var' )
        plt.plot(range(true_y.shape[0]), true_y[:, idx], label='True Var' )
        plt.legend()
        plt.savefig('plots/test_res_' + str(seed) + '_' + str(idx) + '.png')
        plt.clf()
    
#     print('MAE between z and z_hat')
#     compute_rmse(pred_z, true_z)

    # Linear Regression Approximation between z and z_hat
    reg_mat= linear_regression_approx(true_z, pred_z)
    pred_err= np.sqrt( np.mean( (true_z - np.matmul(pred_z, reg_mat))**2 ) )
    print('')
    print('Linear Regression Approximation between z and z_hat')
    print(reg_mat)
    # print('L0 Norm: ', np.sum( np.abs(reg_mat - 0.0001) ) )
#     print('L1 Norm: ', np.linalg.norm(reg_mat, ord=1))
    print('Pred Error: ', pred_err)
    
    key= 'latent_pred_rmse'
    if key not in res.keys():
        res[key]=[]
    res[key].append(pred_err)
        
    # ICA Transformation    
    ica_transform= FastICA()
    z_hat_upd= ica_transform.fit_transform(pred_z)

    reg_mat= linear_regression_approx(true_z, z_hat_upd)
    pred_err= np.sqrt( np.mean( (true_z - np.matmul(z_hat_upd, reg_mat))**2 ) )
    print('')
    print('Post ICA Transformation')
    print(reg_mat)
    # print('L0 Norm: ', np.linalg.norm(reg_mat, ord=1))
#     print('L1 Norm: ', np.linalg.norm(reg_mat, ord=1))
    print('Pred Error: ', pred_err)
    
    key= 'ica_latent_pred_rmse'
    if key not in res.keys():
        res[key]=[]
    res[key].append(pred_err)

print(res)
print('')
print('Final Results')
for key in res.keys():
    res[key]= np.array(res[key])
    print('Metric: ', key, np.mean(res[key]), np.std(res[key]))
