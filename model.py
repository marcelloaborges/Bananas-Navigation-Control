import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, state_size, actions_size, fc1_units=64):
        super().__init__()                
        
        self.fc1 = nn.Linear(state_size, fc1_units)        
        self.fcO = nn.Linear(fc1_units, actions_size)
                
    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fcO(x)
        
        return x
