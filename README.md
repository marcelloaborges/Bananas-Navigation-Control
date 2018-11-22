<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif">


# DRL - Double DQN - Navigation Control
Udacity Deep Reinforcement Learning Nanodegree Program - Navigation Control


### Observations:
- To run the project just execute the <b>main.py</b> file.
- There is also an .ipynb file for jupyter notebook execution.
- If you are not using a windows environment, you will need to download the corresponding <b>"Banana"</b> version for you OS system. Mail me if you need more details about the environment <b>.exe</b> file.
- The <b>checkpoint.pth</b> has the expected average score already hit.


### Requeriments:
- tensorflow: 1.7.1
- Pillow: 4.2.1
- matplotlib
- numpy: 1.11.0
- pytest: 3.2.2
- docopt
- pyyaml
- protobuf: 3.5.2
- grpcio: 1.11.0
- torch: 0.4.1
- pandas
- scipy
- ipykernel
- jupyter: 5.6.0


## The problem:
- The task solved here refers to a navigation control environment where the agent must be able to drive itself 
  collecting yellow bananas while avoiding the blue ones.
- For each yellow banana collected it receives a +1 reward however, when it collects a blue banana the reward is -1.
- The agent can perform 4 actions (left, right, forward or backwards).
- The environment provides information about the state.
- The goal is to get an average score of +13 over 100 consecutive episodes.


## The solution:
- For this problem I used a Double Deep Q-Learning with Experience Replay approach.
- I have checked this approach along with prioritized experience replay but the results were almost the same whereas the performance decreased with the prioritized experience replay implementation due to the fact that the algorithm must have updated the error and the pick change value for all the memory buffer after each backward step. In the tests I did, just tunning the hyperparameters I got almost the same results.
- The random factor on this environment can generate initial sceneries that make the agent converge really fast or take really long. In some tests it could solve the task in less than 200 episodes and in other ones it took almost 400 hundred episodes. The average convergence still occurs around the 200th episode, but in a few cases you can face these wierd sceneries.
- The future goal is to use the Duelling Deep Q-Learning and check how it goes in comparison to the actual solution.
I also want to check the "exploitation vs. exploration" question working with a variable epsilon. For this case
the epsilon is fixed to 5%. I think the convergence ratio may increase working with a decreasing exploration ratio, what I believe would deal better with the sceneries where the agent gets stuck.


### The hyperparameters:
- The file with the hyperparameters configuration is the <b>main.py</b>. 
- If you want you can change the model configuration into the <b>model.py</b> file.
- The actual configuration of the hyperparameters is: 
  - ALPHA: 1
  - GAMMA: 0.99
  - TAU: 1e-3
  - UPDATE_EVERY: 1 
  - BUFFER_SIZE: 1e5
  - BATCH_SIZE: 128
  - Learning Rate: 5e-4

- For the neural model:    
  - Hidden: (state_size, 64)   - ReLU    
  - Output: (64, action_size)  - Linear   
