#Common imports
import sys
import os
import argparse
import random
import copy
import math

import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)
from utils.invertible_network_utils import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Prob ~ Dirichlet(g_i*z) [Class Imbalance] better than Prob ~ Softmax(g_i*z)

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_dim', type=int, default=16, 
                    help='')
parser.add_argument('--num_tasks_list', nargs='+', type=int, default=[8, 12, 16], 
                    help='')
parser.add_argument('--train_size', type=int, default=5000, 
                    help='')
parser.add_argument('--test_size', type=int, default=5000, 
                    help='')
parser.add_argument('--linear_dgp', type=int, default=0,
                    help='')
parser.add_argument('--latent_case', type=str, default='uniform_discrete',
                    help='laplace; uniform')

args = parser.parse_args()
data_dim= args.data_dim
num_tasks_list= args.num_tasks_list

#Random Seed
seed= 10
random.seed(seed*10)
np.random.seed(seed*10) 

#Invertible MLP from the ICML 21 paper

if args.linear_dgp:
    rep_num_layer= 1
else:
    rep_num_layer= 2

rep_net = construct_invertible_mlp(
    n=data_dim,
    n_layers=rep_num_layer,
    act_fct='leaky_relu',
    cond_thresh_ratio=0.0,
    n_iter_cond_thresh=25000 ,
)

print('Invertible MLP to generate x from z')
print(rep_net)

# rep_net= nn.Sequential(
#             nn.Linear(data_dim, data_dim),    
# #             nn.Linear(data_dim, data_dim),    
# #             nn.Linear(data_dim, data_dim),    
#             nn.Sigmoid(),
# )


for num_tasks in num_tasks_list:
    
    #Transformation Functions
    #g = np.random.rand(data_dim, num_tasks)
#     g= np.random.multivariate_normal(np.zeros(data_dim), np.eye(data_dim), size=num_tasks).T
#     g= ortho_group.rvs(data_dim)[:, :num_tasks]
    # Sample orthonormal matrices (scipy.ortho)
    g= np.random.multivariate_normal(np.zeros(data_dim), 10*np.eye(data_dim), size=num_tasks).T
    
    for data_case in ['train', 'val', 'test']:

        print('')
        print('Data Case: ', data_case)

        if data_case == 'train':
            dataset_size= args.train_size
        if data_case == 'val':
            dataset_size= int(args.train_size/4)
        elif data_case == 'test':
            dataset_size= args.test_size    

        z= np.zeros((dataset_size, data_dim))
        for i in range(data_dim):
            if args.latent_case == 'laplace':
                z[:, i]= np.random.laplace(0, 1, dataset_size)
            elif args.latent_case == 'uniform':
                z[:, i]= np.random.uniform(low=0, high=1, size=dataset_size)
            elif args.latent_case == 'uniform_discrete':
                z[:, i]= np.random.randint(2, size= dataset_size)         
    #     z= np.random.multivariate_normal(np.zeros(data_dim), np.eye(data_dim), dataset_size)
    

        print('Latent Z')
        print(np.mean(z[0,:]), np.var(z[0,:]))

        with torch.no_grad():
            x= rep_net( torch.Tensor(z) ).detach().cpu().numpy()
        
        y= np.matmul(z, g)/math.sqrt(data_dim)        
        prob= 1/(1+np.exp(-1*y))
        
        print(prob)
                
        labels=np.zeros((y.shape[0], y.shape[1]))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                labels[i, j]= np.random.binomial(1, prob[i,j], size=1)
                
        pred_labels=np.zeros((y.shape[0], y.shape[1]))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if prob[i,j] >= 0.5:
                    pred_labels[i, j]= 1
                else:
                    pred_labels[i, j]= 0
                    
        print(100*np.mean(labels == pred_labels))
                    
        print('Data Dimensions: ', x.shape, z.shape, y.shape)
        print('Label y')
        print('Class Imbalance in Y')
        print(np.unique(labels, return_counts=True))
    
        
        base_dir= 'data/datasets/multi_bin_classification_' + 'num_layer_'+ str(rep_num_layer) + '_latent_' + args.latent_case + '/'        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir) 

        f= base_dir+ 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' + data_case + '_' + 'x' + '.npy'
        np.save(f, x)

        f= base_dir+ 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' + data_case + '_' + 'z' + '.npy'
        np.save(f, z)

        f= base_dir+ 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' + data_case + '_' + 'y' + '.npy'
        np.save(f, labels)

#         base_dir= 'plots/'
#         plt.scatter(x[:, 0], x[:, 1])
#         plt.savefig(base_dir + 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' +  data_case +  '_x' + '.png')
#         plt.clf()