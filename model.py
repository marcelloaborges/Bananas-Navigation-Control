import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):

    def __init__(self, state_size, actions_size, fc1_units=24):
        super().__init__()                

        self.LR = 5e-4  

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fcO = nn.Linear(fc1_units, actions_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.LR)
        
    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fcO(x)
        
        return x

    def learn(self, value, target):
        loss = F.smooth_l1_loss(value, target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
