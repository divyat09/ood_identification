#Common imports
import sys
import os
import argparse
import random
import copy

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
A = 0.5*np.random.rand(data_dim, data_dim)
# To ensure matrix A is invertible
# A = 0.5*np.dot(A, A.transpose())
g = np.random.rand(data_dim, num_tasks)

for data_case in ['train', 'test']:

    if data_case == 'train':
        dataset_size= args.train_size
    elif data_case == 'test':
        dataset_size= args.test_size    

    z= np.zeros((dataset_size, data_dim))
    for i in range(data_dim):
        z[:, i]= np.random.laplace(0, 1, dataset_size)

    # Sample x and y conditioned on the true latent z
    x= np.matmul(z, A) 
    x= sigmoid(x)
        
#     # Sample the latent variable
#     z= np.random.multivariate_normal(np.zeros(data_dim), np.eye(data_dim), dataset_size)

#     # Sample x and y conditioned on the true latent z
#     x= np.matmul(z, A) 

    y= 50*np.matmul(z, g) + np.random.multivariate_normal(np.zeros(num_tasks), np.eye(num_tasks), dataset_size)

    print('Data Dimensions: ', x.shape, z.shape, y.shape)
    for idx in range(y.shape[1]):
        print(np.mean(y[:,idx]), np.std(y[:, idx]))
    
    base_dir= 'data/datasets/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir) 

    f= base_dir+ data_case + '_' + 'x' + '.npy'
    np.save(f, x)

    f= base_dir+ data_case + '_' + 'z' + '.npy'
    np.save(f, z)

    f= base_dir+ data_case + '_' + 'y' + '.npy'
    np.save(f, y)

    base_dir= 'plots/'
    plt.scatter(x[:, 0], x[:, 1])
    plt.savefig(base_dir + data_case +  '_x' + '.png')
    plt.clf()