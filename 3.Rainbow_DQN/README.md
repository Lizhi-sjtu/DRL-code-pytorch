# Rainbow DQN
This is a concise Pytorch implementation of Rainbow DQN, including Double Q-learning, Dueling network, Noisy network, PER and n-steps Q-learning.<br />

## How to use my code?
You can dircetly run Rainbow_DQN_main.py in your own IDE.<br />

### Trainning environments
You can set the 'env_index' in the code to change the environments.<br />
env_index=0 represent 'CartPole-v1'<br />
env_index=1 represent 'LunarLander-v2'<br />

### How to see the training results?
You can use the tensorboard to visualize the training curves, which are saved in the file 'runs'.<br />
The rewards data are saved as numpy in the file 'data_train'.<br />
The training curves are shown below.<br />
The right picture is smoothed by averaging over a window of 10 steps. The solid line and the shadow respectively represent the average and standard deviation over three different random seeds. (seed=0, 10, 100)<br />
![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.Rainbow_DQN/rainbow_dqn_result.png)

## Reference
[1] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. nature, 2015, 518(7540): 529-533.<br />
[2] Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double q-learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2016, 30(1).<br />
[3] Wang Z, Schaul T, Hessel M, et al. Dueling network architectures for deep reinforcement learning[C]//International conference on machine learning. PMLR, 2016: 1995-2003.<br />
[4] Fortunato M, Azar M G, Piot B, et al. Noisy networks for exploration[J]. arXiv preprint arXiv:1706.10295, 2017.<br />
[5] Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv preprint arXiv:1511.05952, 2015.<br />
[6] Hessel M, Modayil J, Van Hasselt H, et al. Rainbow: Combining improvements in deep reinforcement learning[C]//Thirty-second AAAI conference on artificial intelligence. 2018.<br />
