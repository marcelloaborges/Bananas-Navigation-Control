from unityagents import UnityEnvironment

import numpy as np

import torch

from agent import Agent
from collections import deque

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", no_graphics=False)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents in the environment
print('Number of agents:', len(env_info.agents))
# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# hyperparameters
ALPHA = 1
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 1
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
LR = 5e-4

CHECKPOINT_FOLDER = './'

agent = Agent(
                DEVICE,
                state_size, action_size,
                ALPHA, GAMMA, TAU, UPDATE_EVERY, BUFFER_SIZE, BATCH_SIZE, LR,
                CHECKPOINT_FOLDER
        ) 


def ddqn_train():

    scores = []
    scores_window = deque(maxlen=100)
    n_episodes = 1000
    for episode in range(0, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        
        while True:        
            action = int(agent.act(state, 0.05))
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            
            agent.step(state, action, reward, next_state, done)        

            state = next_state                             # roll over the state to next time step        
            if done:                                       # exit loop if episode finished
                break
        
        agent.checkpoint()

        scores.append(score)
        scores_window.append(score)
        
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, score, np.mean(scores_window)), end="")  

        if np.mean(scores_window) >= 13 and episode >= 100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))                
            break
        
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()    


# train the agent
ddqn_train()

# test the trained agent
agent = Agent(
                DEVICE,
                state_size, action_size,
                ALPHA, GAMMA, TAU, UPDATE_EVERY, BUFFER_SIZE, BATCH_SIZE, LR,
                CHECKPOINT_FOLDER
        ) 

for episode in range(3):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state     
    score = 0                                   

    while True:        
        action = int(agent.act(state))

        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state       
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished       
        score += reward         

        state = next_state                             # roll over the state to next time step        

        if done:                                       # exit loop if episode finished
            break
    
    print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, score))

env.close()