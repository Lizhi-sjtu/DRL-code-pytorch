# PPO-continuous
This is a concise Pytorch implementation of PPO on continuous action space.<br />
There are two alternative probability distributions for actions: Gaussain distribution and Beta distribution, and we provide two files named 'PPO_continuous_Gaussian.py' and 'PPO_continuous_Beta.py' respectively.<br />
The experimental results show that the Beta distribution can achieve better performance than the Gaussian distribution in most cases. <br />

# How to use my code?
You can dircetly run PPO_discrete.py in your own IDE.<br />

## Test environments
You can set the 'env_index' in the codes to change the environments. Here, we test our code in 5 environments.<br />
env_index=0 represent 'Pendulum-v1'<br />
env_index=1 represent 'BipedalWalker-v3'<br />
env_index=2 represent 'HalfCheetah-v2'<br />
env_index=3 represent 'Hopper-v2'<br />
env_index=4 represent 'Walker2d-v2'<br />

## How to see the training results?
You can use the tensorboard to visualize the training curves, which are saved in the file 'runs'.<br />
The rewards data are saved as numpy in the file 'data_train'.<br />
The training curves are shown below,  which are smoothed by averaging over a window of 10 steps.<br />
The solid line and the shadow respectively represent the average and standard deviation over three different random seeds. (seed=0, 10, 100)<br />

![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/Beta_and_Gaussian.png)

# Reference
[1] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.<br />
[2] Schulman J, Moritz P, Levine S, et al. High-dimensional continuous control using generalized advantage estimation[J]. arXiv preprint arXiv:1506.02438, 2015.<br />
[3] Chou P W, Maturana D, Scherer S. Improving stochastic policy gradients in continuous control with deep reinforcement learning using the beta distribution[C]//International conference on machine learning. PMLR, 2017: 834-843.<br />
