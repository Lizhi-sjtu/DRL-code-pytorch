import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def smooth(reward):
    smooth_reward = []
    for i in range(reward.shape[0]):
        if i == 0:
            smooth_reward.append(reward[i])
        else:
            smooth_reward.append(smooth_reward[-1] * 0.9 + reward[i] * 0.1)
    return np.array(smooth_reward)


sns.set_style('darkgrid')

baseline_v0_seed_0 = smooth(np.load('./data_train/REINFORCE_baseline_env_CartPole-v0_number_1_seed_0.npy'))
baseline_v0_seed_10 = smooth(np.load('./data_train/REINFORCE_baseline_env_CartPole-v0_number_1_seed_10.npy'))
baseline_v0_seed_100 = smooth(np.load('./data_train/REINFORCE_baseline_env_CartPole-v0_number_1_seed_100.npy'))

v0_seed_0 = smooth(np.load('./data_train/REINFORCE_env_CartPole-v0_number_1_seed_0.npy'))
v0_seed_10 = smooth(np.load('./data_train/REINFORCE_env_CartPole-v0_number_1_seed_10.npy'))
v0_seed_100 = smooth(np.load('./data_train/REINFORCE_env_CartPole-v0_number_1_seed_100.npy'))

reward_baseline_v0 = np.stack((baseline_v0_seed_0, baseline_v0_seed_10, baseline_v0_seed_100), axis=0)

reward_v0 = np.stack((v0_seed_0, v0_seed_10, v0_seed_100), axis=0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
x = np.arange(baseline_v0_seed_0.shape[0])
sns.tsplot(time=x, data=reward_baseline_v0, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
sns.tsplot(time=x, data=reward_v0, color='dodgerblue', linestyle='-')  # color=darkorange dodgerblue
plt.plot(reward_baseline_v0.mean(),color='darkorange',label='REINFORCE_baseline')
plt.plot(reward_v0.mean(),color='dodgerblue',label='REINFORCE')
plt.title("CartPole-v0", size=16)
plt.xlabel("Steps", size=16)
plt.ylabel("Reward", size=16)
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '20k', '40k', '60k', '80k', '100k'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right')



baseline_v1_seed_0 = smooth(np.load('./data_train/REINFORCE_baseline_env_CartPole-v1_number_1_seed_0.npy'))
baseline_v1_seed_10 = smooth(np.load('./data_train/REINFORCE_baseline_env_CartPole-v1_number_1_seed_10.npy'))
baseline_v1_seed_100 = smooth(np.load('./data_train/REINFORCE_baseline_env_CartPole-v1_number_1_seed_100.npy'))

v1_seed_0 = smooth(np.load('./data_train/REINFORCE_env_CartPole-v1_number_1_seed_0.npy'))
v1_seed_10 = smooth(np.load('./data_train/REINFORCE_env_CartPole-v1_number_1_seed_10.npy'))
v1_seed_100 = smooth(np.load('./data_train/REINFORCE_env_CartPole-v1_number_1_seed_100.npy'))

x = np.arange(baseline_v1_seed_0.shape[0])
reward_baseline_v1 = np.stack((baseline_v1_seed_0, baseline_v1_seed_10, baseline_v1_seed_100), axis=0)
reward_v1 = np.stack((v1_seed_0, v1_seed_10, v1_seed_100), axis=0)

plt.subplot(1, 2, 2)
sns.tsplot(time=x, data=reward_baseline_v1, color='darkorange', linestyle='-')
sns.tsplot(time=x, data=reward_v1, color='dodgerblue', linestyle='-')
plt.plot(reward_baseline_v1.mean(),color='darkorange',label='REINFORCE_baseline')
plt.plot(reward_v1.mean(),color='dodgerblue',label='REINFORCE')
plt.title("CartPole-v1", size=16)
plt.xlabel("Steps", size=16)
plt.ylabel("Reward", size=16)
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '20k', '40k', '60k', '80k', '100k'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right')
plt.show()
