# PPO-continuous
This is a concise Pytorch implementation of PPO on continuous action space with 10 tricks.<br />

## 10 tricks
Trick 1—Advantage Normalization.<br />
Trick 2—State Normalization.<br />
Trick 3 & Trick 4—— Reward Normalization & Reward Scaling.<br />
Trick 5—Policy Entropy.<br />
Trick 6—Learning Rate Decay.<br />
Trick 7—Gradient clip.<br />
Trick 8—Orthogonal Initialization.<br />
Trick 9—Adam Optimizer Epsilon Parameter.<br />
Trick10—Tanh Activation Function.<br />

## How to use my code?
You can dircetly run 'PPO_continuous_main.py' in your own IDE.<br />

## Trainning environments
You can set the 'env_index' in the codes to change the environments. Here, we train our code in 4 environments.<br />
env_index=0 represent 'BipedalWalker-v3'<br />
env_index=1 represent 'HalfCheetah-v2'<br />
env_index=2 represent 'Hopper-v2'<br />
env_index=3 represent 'Walker2d-v2'<br />

## Trainning result
![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/training_result.png)

## Tutorial
If you can read Chinese, you can get more information from this blog.https://zhuanlan.zhihu.com/p/512327050
