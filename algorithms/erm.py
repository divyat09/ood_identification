import os

import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from sklearn.decomposition import FastICA
from utils.metrics import *

from models.fc_regression import FC
from models.invertible_nn import InvertibleNN

class ERM():
    
    def __init__(self, args, train_dataset, val_dataset):
        
        self.args= args
        self.train_dataset= train_dataset
        self.val_dataset= val_dataset
        self.latent_pred_task= args.latent_pred_task
        
        if self.args.invertible_model:
            self.model= InvertibleNN(self.args.data_dim, self.args.num_tasks, self.args.num_layers)
            self.res_dir= 'results/invertible/fc_' + str(self.args.num_layers) + '/'
            
        else:
            self.model= FC(self.args.data_dim, self.args.num_tasks, self.args.num_layers)
            self.res_dir= 'results/non_invertible/fc_' + str(self.args.num_layers) + '/'
        
        print(self.model)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir) 
            
        self.opt= self.get_optimizer(self.args.lr, self.args.weight_decay)     
        
        self.ica_transform= FastICA()
        
        if self.latent_pred_task:
            self.save_path= self.res_dir + 'num_tasks_' + str(self.args.num_tasks) + '_data_dim_' + str(self.args.data_dim) + '_latent_prediction_task.pth'
        else:
            self.save_path= self.res_dir + 'num_tasks_' + str(self.args.num_tasks) + '_data_dim_' + str(self.args.data_dim) + '.pth'
            
    
    def get_optimizer(self, lr, weight_decay):
        
        opt= optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, self.model.parameters()) }, 
                    ], lr= lr, weight_decay= 5e-4, momentum= 0.9,  nesterov=True ) 
        return opt
        
    def save_model(self):
        
        torch.save(self.model.state_dict(), self.save_path)        
        return
    
    def load_model(self):       
        
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        return
    
    def validation(self):
        
        self.model.eval()        
        val_loss=0.0
        count=0
        for batch_idx, (x, y, z) in enumerate(self.val_dataset):
            
            with torch.no_grad():
                
                # Forward Pass
                if self.latent_pred_task:
                    out= self.model.rep_net(x)
                    loss= torch.mean(torch.sum((out-z)**2, dim=1))                    
                else:
                    out= self.model(x)
                    loss= torch.mean(torch.sum((out-y)**2, dim=1))
                    
                val_loss+= loss.item()
                count+=1
        
        self.model.train()    
        return val_loss/count
        
    def train(self):
        
        best_val_score= float('inf')
        best_epoch= -1
        for epoch in range(self.args.num_epochs):
            train_loss=0.0
            count=0
            
            for batch_idx, (x, y, z) in enumerate(self.train_dataset):

                # Forward Pass
                if self.latent_pred_task:
                    out= self.model.rep_net(x)
                    loss= torch.mean(torch.sum((out-z)**2, dim=1))                    
                else:
                    out= self.model(x)
                    loss= torch.mean(torch.sum((out-y)**2, dim=1))
                
                #Backward Pass
                loss.backward()
                
                self.opt.step()
                self.opt.zero_grad()

                train_loss+= loss.item()
                count+=1                
            
            val_score= self.validation()            
            if val_score < best_val_score:
                best_val_score= val_score
                best_epoch= epoch
                self.save_model()                
            
            print('Done Training for Epoch: ', epoch)
            print('MSE Loss: ', train_loss/count)            
            print('Best Epoch: ', best_epoch)
            
        return
        
    def train_ica(self):
        
        # Load the weights of the trained model
        self.load_model()        

        true_z= []
        pred_z= []
        for batch_idx, (x, y, z) in enumerate(self.train_dataset):

            true_z.append(z)
            pred_z.append(self.model.rep_net(x))

        true_z= torch.cat(true_z).detach().numpy()
        pred_z= torch.cat(pred_z).detach().numpy()
        
        # ICA Transformation    
        self.ica_transform.fit(pred_z)

        return