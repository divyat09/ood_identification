import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

class FC(torch.nn.Module):          
     
    def __init__(self, inp_dim, num_tasks):     
        super(FC, self).__init__()
                
        self.inp_dim= inp_dim
        self.num_tasks= num_tasks
        self.rep_net= nn.Sequential(
#                     nn.Linear(inp_dim, inp_dim),    
#                     nn.BatchNorm1d(inp_dim),            
#                     nn.LeakyReLU(0.1),
                    nn.Linear(inp_dim, 100),    
                    nn.BatchNorm1d(100),        
#                     nn.Dropout(),
#                     nn.LeakyReLU(0.2),
            
                    nn.LeakyReLU(0.1),
                    nn.Linear(100, 100),    
                    nn.BatchNorm1d(100),        
            
                    nn.LeakyReLU(0.1),
                    nn.Linear(100, 100),    
                    nn.BatchNorm1d(100),        
            
                    nn.LeakyReLU(0.1),
                    nn.Linear(100, 100),    
                    nn.BatchNorm1d(100),        
            
#                     nn.Dropout(),
#                     nn.LeakyReLU(0.2),
                    nn.LeakyReLU(0.1),
                    nn.Linear(100, inp_dim),
                    nn.BatchNorm1d(inp_dim),
                    nn.LeakyReLU(0.1),
#                     nn.Sigmoid(),
        )
        
        self.fc_net= nn.Sequential(                    
                    nn.Linear(inp_dim, num_tasks)
        )
        
        
    def forward(self, x):        
        out= self.rep_net(x)
        out= self.fc_net(out)
        return out