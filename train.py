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
parser.add_argument('--linear_model', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='non_linear',
                   help='')
parser.add_argument('--inv_reg', type=float, default=1.0,
                   help='Regularizer lambda for invertible model loss: MSE + lambda*Gen_Loss')
parser.add_argument('--lambda', type=float, default=1.0,
                   help='Regularizer lambda for learning ICA: MSE + lambda*ICA_Loss')
parser.add_argument('--final_task', type=int, default=0,
                   help='0: regression; 1: classification')
parser.add_argument('--train_ica', type=int, default=0,
                   help='')
parser.add_argument('--ica_start', type=int, default=20,
                   help='')
parser.add_argument('--train_model', type=int, default=1,
                   help='0: evaluation; 1: training & evaluation')
parser.add_argument('--debug_ica', type=int, default=0,
                   help='')

args = parser.parse_args()
batch_size= args.batch_size
lr= args.lr
num_epochs= args.num_epochs
data_dir= args.data_dir
data_dim= args.data_dim
num_tasks= args.num_tasks
num_seeds= args.num_seeds
invertible_model= args.invertible_model
linear_model= args.linear_model
final_task= args.final_task
train_model= args.train_model

# Load Dataset
kwargs={}
data_obj= BaseDataLoader(data_dir= data_dir, data_case='train', data_dim= data_dim, num_tasks= num_tasks)
train_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

data_obj= BaseDataLoader(data_dir= data_dir, data_case='val', data_dim= data_dim, num_tasks= num_tasks)
val_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

data_obj= BaseDataLoader(data_dir= data_dir, data_case='test', data_dim= data_dim, num_tasks=num_tasks)
test_dataset= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )

res={}

for seed in range(1, 1+num_seeds):
    
    #Seed values
    random.seed(seed*10)
    np.random.seed(seed*10) 
    torch.manual_seed(seed*10)
    
    #Load Algorithm
    method= ERM(args, train_dataset, val_dataset, test_dataset)
    
    #Debug ICA
    if args.debug_ica:
        
        true_x, true_z= get_predictions_check(train_dataset, test_dataset)
        method.train_ica_check()
        ica_x= get_ica_sources(true_x, method.ica_transform)    
        
        score= get_cross_correlation(true_x, true_z)
        print(score)
        
        score= get_cross_correlation(ica_x, true_z)
        print(score)
        
        np.save( 'ica_x_tr.npy', ica_x['tr'] )    
        np.save( 'ica_x_te.npy', ica_x['te'] )    
        
        continue
    
    #Training
    if train_model:
        method.train()
                
    #Test
    method.load_model()    
    
    # When the task is to predict z from x
    if args.latent_pred_task:
        
        true_y, pred_y, true_z, pred_z= get_predictions(method.model, train_dataset, test_dataset, self.args.invertible_model)
        
        #Latent Prediction Error
        rmse,r2= get_latent_prediction_error(pred_z['te'], true_z['te'])   

        key= 'latent_pred_rmse'
        if key not in res.keys():
            res[key]=[]
        res[key].append(rmse)

        key= 'latent_pred_r2'
        if key not in res.keys():
            res[key]=[]
        res[key].append(r2)

        continue
        
    true_y, pred_y, true_z, pred_z= get_predictions(method.model, train_dataset, test_dataset, args.invertible_model) 

    #Label Prediction Error
    if final_task:
        acc= get_direct_prediction_error(pred_y, true_y, final_task= 1)  
        
        key= 'target_pred_acc'
        if key not in res.keys():
            res[key]= []
        res[key].append(acc)        
    else:
        rmse,r2= get_direct_prediction_error(pred_y, true_y)
    
        key= 'target_pred_rmse'
        if key not in res.keys():
            res[key]= []
        res[key].append(rmse)

        key= 'target_pred_r2'
        if key not in res.keys():
            res[key]= []
        res[key].append(r2)
        
    #ICA Transformation
    method.train_ica()
    ica_z= get_ica_sources(pred_z, method.ica_transform)

    #PCA Transformation
    method.train_pca()
    pca_z= get_pca_sources(pred_z, method.pca_transform)

    
    np.save( 'true_z_tr.npy', true_z['tr'] )
    np.save( 'true_z_te.npy', true_z['te'] )
    
    np.save( 'pred_z_tr.npy', pred_z['tr'] )
    np.save( 'pred_z_te.npy', pred_z['te'] )
    
    np.save( 'ica_z_tr.npy', ica_z['tr'] )    
    np.save( 'ica_z_te.npy', ica_z['te'] )    
    

    #Label Prediction Error with ICA
    if final_task:
        acc= get_indirect_prediction_error(ica_z, true_y, final_task= 1)  
        
        key= 'target_ica_pred_acc'
        if key not in res.keys():
            res[key]= []
        res[key].append(acc)        
        
    else:
        rmse,r2= get_indirect_prediction_error(ica_z, true_y)       
        key= 'target_ica_pred_rmse'
        if key not in res.keys():
            res[key]= []
        res[key].append(rmse)
        
        key= 'target_ica_pred_r2'
        if key not in res.keys():
            res[key]= []
        res[key].append(r2)

        
    #Label Prediction Error with PCA
    if final_task:
        acc= get_indirect_prediction_error(pca_z, true_y, final_task= 1)  
        
        key= 'target_pca_pred_acc'
        if key not in res.keys():
            res[key]= []
        res[key].append(acc)        
        
    else:
        rmse,r2= get_indirect_prediction_error(pca_z, true_y)       
        key= 'target_pca_pred_rmse'
        if key not in res.keys():
            res[key]= []
        res[key].append(rmse)
        
        key= 'target_pca_pred_r2'
        if key not in res.keys():
            res[key]= []
        res[key].append(r2)
        
        
    #Latent Covariance Matrix
    score= get_cross_correlation(pred_z, true_z)    
    key= 'latent_pred_score'
    if key not in res.keys():
        res[key]= []
    res[key].append(score)
    
    
    #Latent-ICA Covariance Matrix
    score= get_cross_correlation(ica_z, true_z)
    print(score)
    key= 'ica_latent_pred_score'
    if key not in res.keys():
        res[key]= []
    res[key].append(score)

    
    #Latent-PCA Covariance Matrix
    score= get_cross_correlation(pca_z, true_z)
    print(score)
    key= 'pca_latent_pred_score'
    if key not in res.keys():
        res[key]= []
    res[key].append(score)

    
#     get_mi_score(pred_z, pred_z)    
#     get_mi_score(ica_z, ica_z)
#     get_mi_score(pca_z, pca_z)    
    
#     #First Order Independence Score
#     get_independence_score(pred_z, pred_z)    
#     get_independence_score(ica_z, ica_z)
#     get_independence_score(pca_z, pca_z)    


#     # Plotting RMSE values in label prediction
#     for idx in range(true_y['te'].shape[1]):
# #         plt.plot(range(true_y['te'].shape[0]), pred_y['te'][:, idx], label='Predicted Var' )
# #         plt.plot(range(true_y['te'].shape[0]), true_y['te'][:, idx], label='True Var' )

#         plt.plot(range(50), pred_y['te'][:50, idx], label='Predicted Var' )
#         plt.plot(range(50), true_y['te'][:50, idx], label='True Var' )

#         plt.legend()
#         plt.savefig('plots/test_res_' + 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_seed_' + str(seed) + '_' + str(idx) + '.png')
#         plt.clf()


print('Final Results')
for key in res.keys():
    res[key]= np.array(res[key])
    print('Metric: ', key, np.mean(res[key]), np.std(res[key])/np.sqrt(num_seeds))
