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
from algorithms.erm import ERM
from utils.metrics import *


def compute_rmse(pred_z, z):
    print(np.std(z))
    z= (z - np.mean(z))/ np.std(z)
    pred_z= (pred_z - np.mean(pred_z))/ np.std(pred_z)
        
    print(np.sqrt( np.mean((z - pred_z)**2)) )    


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
parser.add_argument('--weight_decay', type=float, default= 5e-4,
                    help='')
parser.add_argument('--num_epochs', type=int, default= 20,
                    help='')
parser.add_argument('--num_seeds', type=int, default=3,
                    help='')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of FCN layers in representation learning network')
parser.add_argument('--latent_pred_task', type=int, default=0,
                    help='')
parser.add_argument('--invertible_model', type=int, default=0,
                   help='')

args = parser.parse_args()
batch_size= args.batch_size
lr= args.lr
num_epochs= args.num_epochs
data_dim= args.data_dim
num_tasks= args.num_tasks
num_seeds= args.num_seeds
invertible_model= args.invertible_model

# Load Dataset
kwargs={}
data_obj= BaseDataLoader(data_case='train', data_dim= data_dim, num_tasks= num_tasks)
train_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

data_obj= BaseDataLoader(data_case='val', data_dim= data_dim, num_tasks= num_tasks)
val_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

data_obj= BaseDataLoader(data_case='test', data_dim= data_dim, num_tasks=num_tasks)
test_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

res={}

for seed in range(1, 1+num_seeds):
    
    #Seed values
    random.seed(seed*10)
    np.random.seed(seed*10) 
    torch.manual_seed(seed*10)
    
    #Load Algorithm
    method= ERM(args, train_dataset, val_dataset)
    
    #Training
    method.train()
    
    #Test
    method.load_model()
    
    # When the task is to predict z from x
    if args.latent_pred_task:
        
        true_y, pred_y, true_z, pred_z= get_test_predictions(method.model, test_dataset)
        
        #Latent Prediction Error
        rmse,r2= get_latent_prediction_error(pred_z, true_z)   

        key= 'latent_pred_rmse'
        if key not in res.keys():
            res[key]=[]
        res[key].append(rmse)

        key= 'latent_pred_r2'
        if key not in res.keys():
            res[key]=[]
        res[key].append(r2)

        continue
    
    true_y, pred_y, true_z, pred_z= get_test_predictions(method.model, test_dataset)    

    #Label Prediction Error
    rmse,r2= get_label_prediction_error(pred_y, true_y)    
    
    key= 'target_pred_rmse'
    if key not in res.keys():
        res[key]= []
    res[key].append(rmse)
    
    key= 'target_pred_r2'
    if key not in res.keys():
        res[key]= []
    res[key].append(r2)    
    
    #Latent Prediction Error
    rmse,r2= get_latent_prediction_error(pred_z, true_z)   
    
    key= 'latent_pred_rmse'
    if key not in res.keys():
        res[key]=[]
    res[key].append(rmse)

    key= 'latent_pred_r2'
    if key not in res.keys():
        res[key]=[]
    res[key].append(r2)
    
    
    #ICA Transformation
    method.train_ica()
    ica_z= get_ica_sources(pred_z, method.ica_transform) 
    
    #Label Prediction Error with ICA
    rmse,r2= get_label_prediction_error_ica(ica_z, true_y)    
    
    key= 'target_ica_pred_rmse'
    if key not in res.keys():
        res[key]= []
    res[key].append(rmse)
    
    key= 'target_ica_pred_r2'
    if key not in res.keys():
        res[key]= []
    res[key].append(r2)    
    
    #Latent Prediction Error with ICA
    rmse,r2= get_latent_prediction_error(ica_z, true_z) 
    
    key= 'latent_ica_pred_rmse'
    if key not in res.keys():
        res[key]=[]
    res[key].append(rmse)

    key= 'latent_ica_pred_r2'
    if key not in res.keys():
        res[key]=[]
    res[key].append(r2)
        
    # Plotting RMSE values in label prediction
    for idx in range(true_y.shape[1]):
        plt.plot(range(true_y.shape[0]), pred_y[:, idx], label='Predicted Var' )
        plt.plot(range(true_y.shape[0]), true_y[:, idx], label='True Var' )
        plt.legend()
        plt.savefig('plots/test_res_' + 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_seed_' + str(seed) + '_' + str(idx) + '.png')
        plt.clf()
    
    reg_z= linear_regression_approx(true_z, pred_z)
    pred_err= np.sqrt( np.mean( (true_z - reg_z)**2 ) )


print('Final Results')
for key in res.keys():
    res[key]= np.array(res[key])
    print('Metric: ', key, np.mean(res[key]), np.std(res[key]))
