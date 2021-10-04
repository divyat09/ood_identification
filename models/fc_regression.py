import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

class FC(torch.nn.Module):          
     
    def __init__(self, inp_dim, num_tasks, num_layers):     
        super(FC, self).__init__()
                
        self.inp_dim= inp_dim
        self.num_tasks= num_tasks
        self.num_layers= num_layers
                
        self.rep_net= self.get_representation_network()
        
        self.fc_net= nn.Sequential(                    
                    nn.Linear(inp_dim, num_tasks)
        )
        
        
    def forward(self, x):        
        out= self.rep_net(x)
        out= self.fc_net(out)
        return out

    def get_representation_network(self):
    
        modules = []
        modules.append( nn.Linear(self.inp_dim, 100) )
        modules.append( nn.BatchNorm1d(100) )
        modules.append( nn.LeakyReLU(0.1) )
        
        for idx in range(self.num_layers-2):
            modules.append( nn.Linear(100, 100) )
            modules.append( nn.BatchNorm1d(100) )
            modules.append( nn.LeakyReLU(0.1) )
            
        modules.append( nn.Linear(100, self.inp_dim) )
        modules.append( nn.BatchNorm1d(self.inp_dim) )
        modules.append( nn.LeakyReLU(0.1) )
                       
        return nn.Sequential(*modules)        
