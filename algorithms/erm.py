import os
import sys
import math

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

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)
from utils.metrics import *


class ERM():
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset):
        
        self.args= args
        self.train_dataset= train_dataset
        self.val_dataset= val_dataset
        self.test_dataset= test_dataset
        self.latent_pred_task= args.latent_pred_task
        self.inv_reg= args.inv_reg
        self.clf_loss= nn.CrossEntropyLoss()
        
        if self.args.invertible_model:
            
            self.masks= [] 
            for idx in range(self.args.num_layers):
                if idx%2:
                    self.masks.append( int(self.args.data_dim/2)*[0] + int(self.args.data_dim/2)*[1] )
                else:
                    self.masks.append( int(self.args.data_dim/2)*[1] + int(self.args.data_dim/2)*[0] )
                        
            self.model= InvertibleNN(self.args.data_dim, self.args.num_tasks, self.args.num_layers, self.masks)
            self.res_dir= 'results/invertible/fc_' + str(self.args.num_layers) + '/'
            
        else:
            self.model= FC(self.args.data_dim, self.args.num_tasks, self.args.num_layers, self.args.linear_model)
            self.res_dir= 'results/non_invertible/fc_' + str(self.args.num_layers) + '/'
        
        print(self.model)
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir) 
            
        self.opt, self.scheduler= self.get_optimizer(self.args.lr, self.args.weight_decay)  
                
        self.ica_transform= FastICA()
        
        if self.latent_pred_task:
            self.save_path= self.res_dir + 'num_tasks_' + str(self.args.num_tasks) + '_data_dim_' + str(self.args.data_dim) + '_latent_prediction_task.pth'
        else:
            self.save_path= self.res_dir + 'num_tasks_' + str(self.args.num_tasks) + '_data_dim_' + str(self.args.data_dim) + '.pth'
            
    
    def get_optimizer(self, lr, weight_decay):        
                    
        opt= optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, self.model.parameters()) }, 
                    ], lr= lr, weight_decay= 5e-4, momentum= 0.9,  nesterov=True ) 
        
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)        
        return opt, scheduler    
        
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
                    if self.args.invertible_model:
                        out, z_pred, logdet= self.model(x)
                    else:
                        out= self.model(x)
                    
                if self.args.final_task:
                    loss= self.clf_loss(out, y.long())
                else:    
                    loss= torch.mean(((out-y)**2)) 
                    
                val_loss+= loss.item()
                count+=1
        
        self.model.train()    
        return val_loss/count
        
    def train(self):
        
        best_val_score= float('inf')
        best_epoch= -1
        for epoch in range(self.args.num_epochs):            
                
            train_loss=0.0
            train_acc= 0.0
            train_size=0
            count=0
            
            if epoch ==0 and count ==0:
                self.save_model()
                
            self.model.eval()        
            true_y, pred_y, true_z, pred_z= get_predictions(self.model, self.train_dataset, self.test_dataset, self.args.invertible_model)       
            if self.args.final_task:
                acc= get_direct_prediction_error(pred_y, true_y, final_task=1)
                print('Label Prediction Test: ', acc)
            else:
                rmse,r2= get_direct_prediction_error(pred_y, true_y)
                print('Label Prediction Test: ', rmse, r2)
            self.model.train()

            for batch_idx, (x, y, z) in enumerate(self.train_dataset):

                # Forward Pass
                if self.latent_pred_task:
                    out= self.model.rep_net(x)
                    loss= torch.mean((out-z)**2)                    
                else:
                    if self.args.invertible_model:
                        out, z_pred, logdet= self.model(x)
                        x_pred, _= self.model.inverse(z_pred)
#                         gen_loss = torch.log(z_pred.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z_pred**2, -1) - logdet)
                    else:
                        out= self.model(x)

                    if self.args.final_task:
                        loss= self.clf_loss(out, y.long())
                        acc= torch.sum(torch.argmax(out, dim=1) == y)
                    else:    
                        loss= torch.mean(torch.abs(out-y))
#                         loss= torch.mean(((out-y)**2))                        
                
                #Backward Pass
                loss.backward()
                
                self.opt.step()
                self.opt.zero_grad()
                    
                train_loss+= loss.item()
                if self.args.final_task:
                    train_acc+= acc
                train_size+= y.shape[0]
                count+=1                
            
            val_score= self.validation()            
            if val_score < best_val_score:
                best_val_score= val_score
                best_epoch= epoch
                self.save_model()                
            
            print('\n')
            print('Done Training for Epoch: ', epoch)
            print('Loss: ', train_loss/count) 
            if self.args.final_task:
                print('Acc: ', 100*train_acc/train_size)
            print('Best Epoch: ', best_epoch)            
#             print(torch.mean(x_pred-x))            
            
            self.scheduler.step()     
            print(self.scheduler.get_last_lr())
                        
#             rmse,r2= get_indirect_prediction_error(pred_z, true_z, case='train')   
#             print('Latent Prediction Train: ', rmse, r2)            

#             rmse,r2= get_indirect_prediction_error(pred_z, true_z)   
#             print('Latent Prediction Test: ', rmse, r2)            
#             self.model.train()
            
#             self.train_ica()
#             ica_z= get_ica_sources(pred_z, self.ica_transform)

#             rmse,r2= get_indirect_prediction_error(ica_z, true_z, case='train')   
#             print('Latent Prediction ICA Train: ', rmse, r2)            

#             rmse,r2= get_indirect_prediction_error(ica_z, true_z)   
#             print('Latent Prediction ICA Test: ', rmse, r2)            
            
            
        return
        
    def train_ica(self):
        
        # Load the weights of the trained model
        self.load_model()        

        pred_z= []
        for batch_idx, (x, y, z) in enumerate(self.train_dataset):
            pred_z.append(self.model.rep_net(x))

        pred_z= torch.cat(pred_z).detach().numpy()
        
        # ICA Transformation                
        self.ica_transform.fit(pred_z)
        
        return

    
    def train_ica_check(self):
        
        pred_z= []
        for batch_idx, (x, y, z) in enumerate(self.train_dataset):
            pred_z.append(x)

        pred_z= torch.cat(pred_z).detach().numpy()
        
        # ICA Transformation                
        self.ica_transform.fit(pred_z)
        
        return
    
    def get_final_layer_weights(self):
        
        self.load_model()
        
        for p in self.model.fc_net.parameters():
            print(p.data)