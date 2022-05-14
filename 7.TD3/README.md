# TD3
This is a concise Pytorch implementation of TD3(Twin Delayed DDPG) on continuous action space.<br />


## How to use my code?
You can dircetly run 'TD3.py' in your own IDE.<br />

### Trainning environments
You can set the 'env_index' in the codes to change the environments. Here, we train our code in 5 environments.<br />
env_index=0 represent 'Pendulum-v1'<br />
env_index=1 represent 'BipedalWalker-v3'<br />
env_index=2 represent 'HalfCheetah-v2'<br />
env_index=3 represent 'Hopper-v2'<br />
env_index=4 represent 'Walker2d-v2'<br />

### How to see the training results?
You can use the tensorboard to visualize the training curves, which are saved in the file 'runs'.<br />
The rewards data are saved as numpy in the file 'data_train'.<br />
The training curves are shown below,  which are smoothed by averaging over a window of 10 steps.<br />
The solid line and the shadow respectively represent the average and standard deviation over three different random seeds. (seed=0, 10, 100)<br />

![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/7.TD3/TD3_result.png)

## Reference
[1] Fujimoto S, Hoof H, Meger D. Addressing function approximation error in actor-critic methods[C]//International conference on machine learning. PMLR, 2018: 1587-1596.<br />
