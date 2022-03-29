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


env_name = ['CartPole-v0', 'CartPole-v1']


def get_data(env_index, number):
    reward1 = smooth(np.load('./data_train/A2C_env_{}_number_{}_seed_0.npy'.format(env_name[env_index], number)))
    reward2 = smooth(np.load('./data_train/A2C_env_{}_number_{}_seed_10.npy'.format(env_name[env_index], number)))
    reward3 = smooth(np.load('./data_train/A2C_env_{}_number_{}_seed_100.npy'.format(env_name[env_index], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len

def get_data2(env_index, number):
    reward1 = np.load('./data_train/A2C_env_{}_number_{}_seed_0.npy'.format(env_name[env_index], number))
    reward2 = np.load('./data_train/A2C_env_{}_number_{}_seed_10.npy'.format(env_name[env_index], number))
    reward3 = np.load('./data_train/A2C_env_{}_number_{}_seed_100.npy'.format(env_name[env_index], number))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len

sns.set_style('darkgrid')
plt.figure()

plt.subplot(1, 2, 1)
CPV0_number2, len = get_data(env_index=0, number=9)
sns.tsplot(time=np.arange(len), data=CPV0_number2, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.title('CartPole-v0', size=14)
plt.xlabel("Steps", size=14)
plt.ylabel("Reward", size=14)
plt.xticks([0, 100, 200, 300], ['0', '100K', '200K', '300K'], size=12)
plt.yticks(size=12)

plt.subplot(1, 2, 2)
CPV1_number2, len = get_data(env_index=1, number=9)
sns.tsplot(time=np.arange(270), data=CPV1_number2[:,:270], color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.title("CartPole-v1", size=14)
plt.xlabel("Steps", size=14)
plt.ylabel("Reward", size=14)
plt.xticks([0, 100, 200, 270], ['0', '100K', '200K', '270K'], size=12)
plt.yticks(size=12)

plt.show()
