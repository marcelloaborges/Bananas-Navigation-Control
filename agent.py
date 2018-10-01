import torch
import random
import numpy as np
from model import Model
from replay_memory import ReplayMemory

class Agent():

    def __init__(self, state_size, actions_size, fc1_units=24):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.ALPHA = 1
        self.GAMMA = 0.99
        self.TAU = 0.001
        self.UPDATE_EVERY = 5
        self.BUFFER_SIZE = int(10000)
        self.BATCH_SIZE = 300

        self.state_size = state_size
        self.actions_size = actions_size

        self.model = Model(state_size, actions_size, fc1_units).to(self.device)
        self.target_model = Model(state_size, actions_size, fc1_units).to(self.device)
        
        self.memory = ReplayMemory(self.BUFFER_SIZE, self.BATCH_SIZE, self.device)

        self.t_step = 0
    
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        
        self.t_step += 1
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
        
        self.model.learn(Q_value, Q_target)

        # update target model
        self.soft_update_target_model()

    def soft_update_target_model(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)