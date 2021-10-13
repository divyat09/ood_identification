import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

class FC(torch.nn.Module):          
     
    def __init__(self, inp_dim, num_tasks, num_layers, linear_model):     
        super(FC, self).__init__()
                
        self.inp_dim= inp_dim
        self.num_tasks= num_tasks
        self.num_layers= num_layers
        self.linear_model= linear_model
                
        self.rep_net= self.get_representation_network()
        
        self.fc_net= nn.Sequential(                    
#                     nn.Linear(100, num_tasks),
                    nn.Linear(inp_dim, num_tasks),
        )
        
        
    def forward(self, x):        
        out= self.rep_net(x)
        out= self.fc_net(out)
        return out

    def get_representation_network(self):
    
        modules = []
        hid_dim= 100
        if self.linear_model:
            #Question: Does batch-norm introduce non-linearity into the model (Should I avoid using it for single FC linear case)
            modules.append( nn.Linear(self.inp_dim, self.inp_dim) )
        else:      
            modules.append( nn.Linear(self.inp_dim, hid_dim) )
            modules.append( nn.BatchNorm1d(hid_dim) )
            modules.append( nn.LeakyReLU(0.5) )

            for idx in range(self.num_layers-2):
                modules.append( nn.Linear(hid_dim, hid_dim) )
                modules.append( nn.BatchNorm1d(hid_dim) )
                modules.append( nn.LeakyReLU(0.5) )

            modules.append( nn.Linear(hid_dim, self.inp_dim) )
            modules.append( nn.BatchNorm1d(self.inp_dim) )
            modules.append( nn.LeakyReLU(0.5) )
                       
        return nn.Sequential(*modules)        
