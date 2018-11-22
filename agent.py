import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import os

from model import Model

from replay_memory import ReplayMemory

class Agent():

    def __init__(self, 
            device,
            state_size, actions_size,
            alpha, gamma, TAU, update_every, buffer_size, batch_size, LR,
            CHECKPOINT_FOLDER = './'):
        
        self.DEVICE = device

        self.state_size = state_size
        self.actions_size = actions_size

        self.ALPHA = alpha
        self.GAMMA = gamma
        self.TAU = TAU
        self.UPDATE_EVERY = update_every
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.LR = LR

        self.CHECKPOINT_FOLDER = CHECKPOINT_FOLDER
    

        self.model = Model(state_size, actions_size).to(self.DEVICE)
        self.target_model = Model(state_size, actions_size).to(self.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        
        if os.path.isfile('checkpoint.pth'):
            self.model.load_state_dict(torch.load('checkpoint.pth'))
            self.target_model.load_state_dict(torch.load('checkpoint.pth'))

        self.memory = ReplayMemory(self.BUFFER_SIZE, self.BATCH_SIZE, self.DEVICE)

        self.t_step = 0
    
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
                
        if np.random.uniform() < eps:
            return random.choice(np.arange(self.actions_size))            
        else:
            action = np.argmax(action_values.cpu().data.numpy())
            return action
            
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):        
        states, actions, rewards, next_states, dones = experiences        

        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = self.ALPHA * (rewards + self.GAMMA * Q_targets_next * (1 - dones))

        Q_value = self.model(states).gather(1, actions)
                
        loss = F.smooth_l1_loss(Q_value, Q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target model
        self.soft_update_target_model()

    def soft_update_target_model(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):        
        torch.save(self.model.state_dict(), self.CHECKPOINT_FOLDER + 'checkpoint.pth')