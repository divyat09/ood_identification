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

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)
from utils.invertible_network_utils import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_dim', type=int, default=2, 
                    help='')
parser.add_argument('--num_tasks_list', nargs='+', type=int, default=[1, 8, 32, 64, 128], 
                    help='')
parser.add_argument('--train_size', type=int, default=5000, 
                    help='')
parser.add_argument('--test_size', type=int, default=100, 
                    help='')

args = parser.parse_args()
data_dim= args.data_dim
num_tasks_list= args.num_tasks_list


#Random Seed
seed= 10
random.seed(seed*10)
np.random.seed(seed*10) 

# Invertible MLP from the ICML 21 paper
rep_net = construct_invertible_mlp(
    n=data_dim,
    n_layers=2,
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
    g = np.random.rand(data_dim, num_tasks)
    
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
            z[:, i]= np.random.laplace(0, 1, dataset_size)
    #     z= np.random.multivariate_normal(np.zeros(data_dim), np.eye(data_dim), dataset_size)

        print('Latent Z')
        print(np.mean(z[0,:]), np.var(z[0,:]))

        with torch.no_grad():
            x= rep_net( torch.Tensor(z) ).detach().cpu().numpy()

        print('Data X SVD')
        print( np.linalg.svd( np.matmul(x.T, x) )[1] )

        y= 50*np.matmul(z, g)/math.sqrt(data_dim) + np.random.multivariate_normal(np.zeros(num_tasks), 10*np.eye(num_tasks), dataset_size)

        print('Data Dimensions: ', x.shape, z.shape, y.shape)
        print('Label y')
        for idx in range(y.shape[1]):
            print(np.mean(y[:,idx]), np.std(y[:, idx]))

        base_dir= 'data/datasets/'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir) 

        f= base_dir+ 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' + data_case + '_' + 'x' + '.npy'
        np.save(f, x)

        f= base_dir+ 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' + data_case + '_' + 'z' + '.npy'
        np.save(f, z)

        f= base_dir+ 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' + data_case + '_' + 'y' + '.npy'
        np.save(f, y)

        base_dir= 'plots/'
        plt.scatter(x[:, 0], x[:, 1])
        plt.savefig(base_dir + 'tasks_' + str(num_tasks) + '_dim_' + str(data_dim) + '_' +  data_case +  '_x' + '.png')
        plt.clf()