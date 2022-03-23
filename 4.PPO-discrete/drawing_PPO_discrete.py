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


env_name = ['CartPole-v1', 'LunarLander-v2']


def get_data(env_index, number):
    reward1 = smooth(np.load('./data_train/PPO_discrete_env_{}_number_{}_seed_0.npy'.format(env_name[env_index], number)))
    reward2 = smooth(np.load('./data_train/PPO_discrete_env_{}_number_{}_seed_10.npy'.format(env_name[env_index], number)))
    reward3 = smooth(np.load('./data_train/PPO_discrete_env_{}_number_{}_seed_100.npy'.format(env_name[env_index], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len


sns.set_style('darkgrid')
plt.figure()

plt.subplot(1, 2, 1)
CP_number2, len = get_data(env_index=0, number=2)
sns.tsplot(time=np.arange(len), data=CP_number2, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.title('CartPole-v1', size=14)
plt.xlabel("Steps", size=14)
plt.ylabel("Reward", size=14)
plt.xticks([0, 20, 40, 60], ['0', '100k', '200k', '300k'], size=12)
plt.yticks(size=12)

plt.subplot(1, 2, 2)
LL_number2, len = get_data(env_index=1, number=2)
sns.tsplot(time=np.arange(len), data=LL_number2, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.title("LunarLander-v2", size=14)
plt.xlabel("Steps", size=14)
plt.ylabel("Reward", size=14)
plt.xticks([0, 20, 40, 60, 80,100], ['0', '100k', '200k', '300k', '400k','500k'], size=12)
plt.yticks(size=12)

plt.show()
