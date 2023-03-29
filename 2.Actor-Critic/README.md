# Actor-Critic(A2C)
This is a concise Pytorch implementation of Advantage Actor-Critic(A2C).<br />

## How to use my code?
You can dircetly run A2C.py in your own IDE.<br />

### Trainning environments
You can set the 'env_index' in the codes to change the environments.<br />
env_index=0 represent 'CartPole-v0'<br />
env_index=1 represent 'CartPole-v1'<br />

### How to see the training results?
You can use the tensorboard to visualize the training curves, which are saved in the file 'runs'.<br />
The rewards data are saved as numpy in the file 'data_train'.<br />
The training curves are shown below,  which are smoothed by averaging over a window of 10 steps.<br />
The solid line and the shadow respectively represent the average and standard deviation over three different random seeds. (seed=0, 10, 100)<br />
![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/2.Actor-Critic/A2C_results.png)
