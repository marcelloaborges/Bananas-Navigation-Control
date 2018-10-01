from unityagents import UnityEnvironment
import numpy as np
from agent import Agent
from collections import deque
import torch
import os

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", no_graphics=True)


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


agent = Agent(state_size, action_size)
if os.path.isfile('checkpoint.pth'):
    agent.model.load_state_dict(torch.load('checkpoint.pth'))
    agent.target_model.load_state_dict(torch.load('checkpoint.pth'))

scores_window = deque(maxlen=100)
for i in range(0, 2000):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    
    while True:
        # action = np.random.randint(action_size)        # select an action
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
    
    scores_window.append(score)
    avarage_score = np.mean(scores_window)
    if avarage_score >= 13 and len(scores_window) == 100:        
        print("Solved! Episode: ", i, " => Avarage Score: ", "{:10.2f}".format(avarage_score), "|Score: ", "{:10.2f}".format(score))
        torch.save(agent.model.state_dict(), 'checkpoint.pth')      
        break
    
    print("Episode: ", i, " => Avarage Score: ", "{:10.2f}".format(avarage_score), "|Score: ", "{:10.2f}".format(score))
    torch.save(agent.model.state_dict(), 'checkpoint.pth')        

env.close()