import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

import utils.encoders as encoders

class InvertibleNN(torch.nn.Module):          
     
    def __init__(self, inp_dim, num_tasks, num_layers):     
        super(InvertibleNN, self).__init__()
                
        self.inp_dim= inp_dim
        self.num_tasks= num_tasks
        self.num_layers= num_layers
        self.rep_net=  encoders.get_mlp(
                n_in=inp_dim,
                n_out=inp_dim,
                layers= self.num_layers*[
                    100,
                ],
                output_normalization=None,
            )

        
        self.fc_net= nn.Sequential(                    
                    nn.Linear(inp_dim, num_tasks)
        )
        
        
    def forward(self, x):        
        out= self.rep_net(x)
        out= self.fc_net(out)
        return out


class FC_ICA(torch.nn.Module):          
     
    def __init__(self, inp_dim, num_tasks):     
        super(FC_ICA, self).__init__()
                
        self.inp_dim= inp_dim
        self.num_tasks= num_tasks
        
        self.fc_net= nn.Sequential(                    
                    nn.Linear(inp_dim, num_tasks)
        )
        
        
    def forward(self, x):        
        out= self.fc_net(x)
        return out