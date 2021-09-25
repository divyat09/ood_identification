import os
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class BaseDataLoader(data_utils.Dataset):
    def __init__(self, data_case='train', data_dim=32, num_tasks=1):
        self.data_dir = 'data/datasets/'
        self.data_case = data_case
        self.num_tasks = num_tasks
        self.data_dim = data_dim
        
        self.data, self.labels, self.latents= self.load_data(self.data_dir, self.data_case)
        
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        z = self.latents[index]
            
        return x, y, z
    
    def load_data(self, data_dir, data_case):
        x= np.load(data_dir + 'tasks_' + num_tasks + '_dim_' + str(data_dim) +  '_' + data_case +  '_' + 'x' + '.npy')
        y= np.load(data_dir + 'tasks_' + num_tasks + '_dim_' + str(data_dim) +  '_' + data_case +  '_' + 'y' + '.npy')
        z= np.load(data_dir + 'tasks_' + num_tasks + '_dim_' + str(data_dim) +  '_' +  data_case +  '_' + 'z' + '.npy')
        
        x= torch.tensor(x).float()
        y= torch.tensor(y).float()
        z= torch.tensor(z).float()
        
        return x, y, z