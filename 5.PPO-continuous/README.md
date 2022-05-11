# PPO-continuous
This is a concise Pytorch implementation of PPO on continuous action space with 10 tricks.<br />

# 10 tricks
Trick 1—Advantage Normalization
Trick 2—State Normalization
Trick 3 & Trick 4—— Reward Normalization & Reward Scaling
Trick 5—Policy Entropy
Trick 6—Learning Rate Decay
Trick 7—Gradient clip
Trick 8—Orthogonal Initialization
Trick 9—Adam Optimizer Epsilon Parameter
Trick10—Tanh Activation Function

## Test environments
You can set the 'env_index' in the codes to change the environments. Here, we test our code in 5 environments.<br />
env_index=0 represent 'BipedalWalker-v3'<br />
env_index=1 represent 'HalfCheetah-v2'<br />
env_index=2 represent 'Hopper-v2'<br />
env_index=3 represent 'Walker2d-v2'<br />
