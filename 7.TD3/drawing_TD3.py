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


env_name = ['Pendulum-v1', 'BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']


def get_data(env_index, number):
    reward1 = smooth(np.load('./data_train/TD3_env_{}_number_{}_seed_0.npy'.format(env_name[env_index], number)))
    reward2 = smooth(np.load('./data_train/TD3_env_{}_number_{}_seed_10.npy'.format(env_name[env_index], number)))
    reward3 = smooth(np.load('./data_train/TD3_env_{}_number_{}_seed_100.npy'.format(env_name[env_index], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len


def get_data2(env_index, number):
    reward1 = smooth(np.load('./data_train/TD3_env_{}_number_{}_seed_0.npy'.format(env_name[env_index], number))[0:200])
    reward2 = smooth(np.load('./data_train/TD3_env_{}_number_{}_seed_10.npy'.format(env_name[env_index], number))[0:200])
    reward3 = smooth(np.load('./data_train/TD3_env_{}_number_{}_seed_100.npy'.format(env_name[env_index], number))[0:200])
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len


sns.set_style('darkgrid')
plt.figure()

plt.subplot(2, 3, 1)
PV_number1, len = get_data(env_index=0, number=1)
sns.tsplot(time=np.arange(len), data=PV_number1, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(PV_number1.mean(0), color='darkorange', label='TD3')
plt.title("Pendulum-v1", size=12)
plt.xlabel("Steps", size=12)
plt.ylabel("Reward", size=12)
plt.xticks([0, 20, 40, 60], ['0', '100K', '200K', '300K'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right', fontsize=14)

plt.subplot(2, 3, 2)
BW_number1, len = get_data2(env_index=1, number=1)
sns.tsplot(time=np.arange(len), data=BW_number1, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(BW_number1.mean(0), color='darkorange', label='TD3')
plt.title("BipedalWalker-v3", size=12)
plt.xlabel("Steps", size=12)
plt.ylabel("Reward", size=12)
plt.xticks([0, 40, 80, 120, 160, 200], ['0', '200K', '400K', '600K', '800K', '1M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right', fontsize=14)

plt.subplot(2, 3, 3)
PV_number1, len = get_data(env_index=2, number=1)
sns.tsplot(time=np.arange(len), data=PV_number1, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(PV_number1.mean(0), color='darkorange', label='TD3')
plt.title("HalfCheetah-v2", size=12)
plt.xlabel("Steps", size=12)
plt.ylabel("Reward", size=12)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['0', '0.5M', '1.0M', '1.5M', '2M', '2.5M', '3M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right', fontsize=14)

plt.subplot(2, 3, 4)
Hopper_number1, len = get_data(env_index=3, number=1)
sns.tsplot(time=np.arange(len), data=Hopper_number1, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(Hopper_number1.mean(0), color='darkorange', label='TD3')
plt.title("Hopper-v2", size=12)
plt.xlabel("Steps", size=12)
plt.ylabel("Reward", size=12)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['0', '0.5M', '1.0M', '1.5M', '2M', '2.5M', '3M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right', fontsize=14)

plt.subplot(2, 3, 5)
Walker_number1, len = get_data(env_index=4, number=1)
sns.tsplot(time=np.arange(len), data=Walker_number1, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(Walker_number1.mean(0), color='darkorange', label='TD3')
plt.title("Walker2d-v2", size=12)
plt.xlabel("Steps", size=12)
plt.ylabel("Reward", size=12)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['0', '0.5M', '1.0M', '1.5M', '2M', '2.5M', '3M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right', fontsize=14)

plt.show()
