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


def get_data(DQN_name, env_index, number):
    reward1 = smooth(np.load('./data_train/{}_env_{}_number_{}_seed_0.npy'.format(DQN_name, env_name[env_index], number))[0:200])
    reward2 = smooth(np.load('./data_train/{}_env_{}_number_{}_seed_10.npy'.format(DQN_name, env_name[env_index], number)))
    reward3 = smooth(np.load('./data_train/{}_env_{}_number_{}_seed_100.npy'.format(DQN_name, env_name[env_index], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len


def smooth2(reward):
    smooth_reward = []
    for i in range(reward.shape[0]):
        if i == 0:
            smooth_reward.append(reward[i])
        else:
            smooth_reward.append(smooth_reward[-1] * 0.5 + reward[i] * 0.5)
    return np.array(smooth_reward)

def get_data2(DQN_name, env_index, number):
    reward = np.load('./data_train/{}_env_{}_number_{}_seed_0.npy'.format(DQN_name, env_name[env_index], number))[0:90]

    return reward



sns.set_style('darkgrid')
plt.figure()

plt.subplot(1, 2, 1)
DQN_CP_4 = get_data2(DQN_name='DQN', env_index=0, number=4)
plt.plot(DQN_CP_4, color='darkorange', label='DQN')

DDQN_CP_4= get_data2(DQN_name='DDQN', env_index=0, number=4)
# sns.tsplot(time=np.arange(T), data=DDQN_CP_1[:,:T], color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(DDQN_CP_4, color='dodgerblue', label='DDQN')

D3QN_CP_4 = get_data2(DQN_name='D3QN', env_index=0, number=4)
# sns.tsplot(time=np.arange(T), data=D3QN_CP_1[:,:T], color='g', linestyle='-')  # color=darkorange dodgerblue
plt.plot(D3QN_CP_4, color='r', label='D3QN')

plt.title('CartPole-v1', size=16)
plt.xlabel("Steps", size=16)
plt.ylabel("Reward", size=16)
plt.xticks([0, 30,60,90], ['0', '30K','60K','90K'], size=14)
plt.yticks(size=14)
plt.legend(loc='lower right',fontsize=14)

plt.subplot(1, 2, 2)
DQN_LL_4, len = get_data(DQN_name='DQN', env_index=1, number=4)
sns.tsplot(time=np.arange(len), data=DQN_LL_4, color='darkorange', linestyle='-')  # color=darkorange dodgerblue
plt.plot(DQN_LL_4.mean(0), color='darkorange', label='DQN')
plt.legend(loc='lower right')

DDQN_CP_4, len = get_data(DQN_name='DDQN', env_index=1, number=4)
sns.tsplot(time=np.arange(len), data=DDQN_CP_4, color='dodgerblue', linestyle='-')  # color=darkorange dodgerblue
plt.plot(DDQN_CP_4.mean(0), color='dodgerblue', label='DDQN')
plt.legend(loc='lower right')

D3QN_CP_4, len = get_data(DQN_name='D3QN', env_index=1, number=4)
sns.tsplot(time=np.arange(len), data=D3QN_CP_4, color='r', linestyle='-')  # color=darkorange dodgerblue
plt.plot(D3QN_CP_4.mean(0), color='r', label='D3QN')

plt.title('LunarLander-v2', size=16)
plt.xlabel("Steps", size=16)
plt.ylabel("Reward", size=16)
plt.xticks([0, 50, 100, 150, 200], ['0', '50K', '100K', '150K', '200K'], size=14)
plt.yticks(size=14)
plt.legend(loc='lower right',fontsize=14)

plt.show()
