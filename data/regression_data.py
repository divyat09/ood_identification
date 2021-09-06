#Common imports
import sys
import os
import argparse
import random
import copy

import numpy as np
from scipy import stats

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_dim', type=int, default=2, 
                    help='')
parser.add_argument('--num_tasks', type=int, default=1, 
                    help='')
parser.add_argument('--train_size', type=int, default=1000, 
                    help='')
parser.add_argument('--test_size', type=int, default=100, 
                    help='')

args = parser.parse_args()
data_dim= args.data_dim
num_tasks= args.num_tasks

#Random Seed
seed= 10
random.seed(seed*10)
np.random.seed(seed*10) 

#Transformation Functions
A = np.random.rand(data_dim, data_dim)
# To ensure matrix A is invertible
A = 5*np.dot(A, A.transpose())
g = np.random.rand(data_dim, num_tasks)

for data_case in ['train', 'test']:

    if data_case == 'train':
        dataset_size= args.train_size
    elif data_case == 'test':
        dataset_size= args.test_size    

    # Sample the latent variable
    z= np.random.multivariate_normal(np.zeros(data_dim), np.eye(data_dim), dataset_size)

    # Sample x and y conditioned on the true latent z
    x= np.matmul(z, A) 
    y= 10 + np.matmul(z, g) + np.random.multivariate_normal(np.zeros(num_tasks), np.eye(num_tasks), dataset_size)
#     y= 10 + np.matmul(z, g)

    print('Data Dimensions: ', x.shape, z.shape, y.shape)

    base_dir= 'data/datasets/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir) 

    f= base_dir+ data_case + '_' + 'x' + '.npy'
    np.save(f, x)

    f= base_dir+ data_case + '_' + 'z' + '.npy'
    np.save(f, z)

    f= base_dir+ data_case + '_' + 'y' + '.npy'
    np.save(f, y)

