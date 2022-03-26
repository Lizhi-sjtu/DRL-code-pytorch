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
    reward1 = smooth(np.load('./data_train/PPO_continuous_Beta_env_{}_number_{}_seed_0.npy'.format(env_name[env_index], number)))
    reward2 = smooth(np.load('./data_train/PPO_continuous_Beta_env_{}_number_{}_seed_10.npy'.format(env_name[env_index], number)))
    reward3 = smooth(np.load('./data_train/PPO_continuous_Beta_env_{}_number_{}_seed_100.npy'.format(env_name[env_index], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len

def get_data2(env_index, number):
    reward1 = smooth(np.load('./data_train/PPO_continuous_Gaussian_env_{}_number_{}_seed_0.npy'.format(env_name[env_index], number)))
    reward2 = smooth(np.load('./data_train/PPO_continuous_Gaussian_env_{}_number_{}_seed_10.npy'.format(env_name[env_index], number)))
    reward3 = smooth(np.load('./data_train/PPO_continuous_Gaussian_env_{}_number_{}_seed_100.npy'.format(env_name[env_index], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len

sns.set_style('darkgrid')
plt.figure()

T=180
plt.subplot(2, 3, 1)
HC_Beta_number2, len = get_data(env_index=0, number=2)
sns.tsplot(time=np.arange(T), data=HC_Beta_number2[:,:T], color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(HC_Beta_number2[:,:T].mean(0), color='darkorange', label='Beta')
BW_Gaussian_number1, len = get_data2(env_index=0, number=1)
sns.tsplot(time=np.arange(T), data=BW_Gaussian_number1[:,:T], color='dodgerblue', linestyle='-')  # color=darkorange dodgerblue
plt.plot(BW_Gaussian_number1[:,:T].mean(0), color='dodgerblue', label='Gaussian')
plt.title("Pendulum-v1", size=13)
plt.xlabel("Steps", size=13)
plt.ylabel("Reward", size=13)
plt.xticks([0, 60, 120, 180], ['0', '300K', '600K', '900K'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right',fontsize=14)

plt.subplot(2, 3, 2)
BW_Beta_number2, len = get_data(env_index=1, number=2)
sns.tsplot(time=np.arange(len), data=BW_Beta_number2, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(BW_Beta_number2.mean(0), color='darkorange', label='Beta')
BW_Gaussian_number1, len = get_data2(env_index=1, number=1)
sns.tsplot(time=np.arange(len), data=BW_Gaussian_number1, color='dodgerblue', linestyle='-')  # color=darkorange dodgerblue
plt.plot(BW_Gaussian_number1.mean(0), color='dodgerblue', label='Gaussian')
plt.title("BipedalWalker-v3", size=13)
plt.xlabel("Steps", size=13)
plt.ylabel("Reward", size=13)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['0', '0.5M', '1.0M', '1.5M', '2M', '2.5M', '3M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right',fontsize=14)

plt.subplot(2, 3, 3)
HC_Beta_number2, len = get_data(env_index=2, number=2)
sns.tsplot(time=np.arange(len), data=HC_Beta_number2, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(HC_Beta_number2.mean(0), color='darkorange', label='Beta')
HC_Gaussian_number1, len = get_data2(env_index=2, number=1)
sns.tsplot(time=np.arange(len), data=HC_Gaussian_number1, color='dodgerblue', linestyle='-')  # color=darkorange dodgerblue
plt.plot(HC_Gaussian_number1.mean(0), color='dodgerblue', label='Gaussian')
plt.title("HalfCheetah-v2", size=13)
plt.xlabel("Steps", size=13)
plt.ylabel("Reward", size=13)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['0', '0.5M', '1.0M', '1.5M', '2M', '2.5M', '3M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right',fontsize=14)

plt.subplot(2, 3, 4)
Hopper_Beta_number2, len = get_data(env_index=3, number=2)
sns.tsplot(time=np.arange(400), data=Hopper_Beta_number2[:,:400], color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(Hopper_Beta_number2[:,:400].mean(0), color='darkorange', label='Beta')
Hopper_Gaussian_number1, len = get_data2(env_index=3, number=1)
sns.tsplot(time=np.arange(400), data=Hopper_Gaussian_number1[:,:400], color='dodgerblue', linestyle='-')  # color=darkorange dodgerblue
plt.plot(Hopper_Gaussian_number1[:,:400].mean(0), color='dodgerblue', label='Gaussian')
plt.title("Hopper-v2", size=13)
plt.xlabel("Steps", size=13)
plt.ylabel("Reward", size=13)
plt.xticks([0, 100, 200, 300, 400], ['0', '0.5M', '1.0M', '1.5M', '2M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right',fontsize=14)

plt.subplot(2, 3, 5)
Walker_Beta_number2, len = get_data(env_index=4, number=2)
sns.tsplot(time=np.arange(len), data=Walker_Beta_number2, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(Walker_Beta_number2.mean(0), color='darkorange', label='Beta')
Walker_Gaussian_number1, len = get_data2(env_index=4, number=1)
sns.tsplot(time=np.arange(len), data=Walker_Gaussian_number1, color='dodgerblue', linestyle='-')  # color=darkorange dodgerblue
plt.plot(Walker_Gaussian_number1.mean(0), color='dodgerblue', label='Gaussian')
plt.title("Walker2d-v2", size=13)
plt.xlabel("Steps", size=13)
plt.ylabel("Reward", size=13)
plt.xticks([0, 100, 200, 300, 400, 500, 600], ['0', '0.5M', '1.0M', '1.5M', '2M', '2.5M', '3M'], size=12)
plt.yticks(size=12)
plt.legend(loc='lower right',fontsize=14)

plt.show()