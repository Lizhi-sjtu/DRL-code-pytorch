# DQN
This is a concise Pytorch implementation of DQN, including original DQN, Double-DQN(DDQN), Dueling Double DQN(D3QN).<br />
m
## How to use my code?
You can dircetly run DQN_DDQN_D3QN.py in your own IDE.<br />
In our code, 'if_double' means whether to use 'double Q-network', 'if_dueling' means whether to use 'dueling network'.<br />
If 'if_double=True' and 'if_dueling=True', then we will use the Dueling Double DQN(D3QN).<br />
If 'if_double=True' and 'if_dueling=False', then we will use the Double DQN(DDQN).<br />
If 'if_double=False' and 'if_dueling=False', then we will use the original DQN.<br />

### Test environments
You can set the 'env_index' in the code to change the environments.<br />
env_index=0 represent 'CartPole-v1'<br />
env_index=1 represent 'LunarLander-v2'<br />

### How to see the training results?
You can use the tensorboard to visualize the training curves, which are saved in the file 'runs'.<br />
The rewards data are saved as numpy in the file 'data_train'.<br />
The training curves are shown below.<br />
The right picture is smoothed by averaging over a window of 10 steps. The solid line and the shadow respectively represent the average and standard deviation over three different random seeds. (seed=0, 10, 100)<br />
![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.DQN/DQN.png)

## Reference
[1] Mnih V, Kavukcuoglu K, Silver D, et al. Playing atari with deep reinforcement learning[J]. arXiv preprint arXiv:1312.5602, 2013.<br />
[2] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. nature, 2015, 518(7540): 529-533.<br />
[3] Wang Z, Schaul T, Hessel M, et al. Dueling network architectures for deep reinforcement learning[C]//International conference on machine learning. PMLR, 2016: 1995-2003.
