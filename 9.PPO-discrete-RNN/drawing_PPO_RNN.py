import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def smooth(reward):
    smooth_reward = []
    for i in range(reward.shape[0]):
        if i == 0:
            smooth_reward.append(reward[i])
        else:
            smooth_reward.append(smooth_reward[-1] * 0.8 + reward[i] * 0.2)
    return np.array(smooth_reward)


env_name = ['CartPole-v1', 'LunarLander-v2']
colors = ["r", 'darkorange', 'dodgerblue', 'aqua', 'limegreen', 'magenta', 'chocolate', 'indigo', 'gray', 'aqua', 'g']


def drawing_CP(plt, number, color, label):
    reward1 = smooth(np.load('./data_train/PPO_env_{}_number_{}_seed_0.npy'.format(env_name[0], number)))
    reward2 = smooth(np.load('./data_train/PPO_env_{}_number_{}_seed_10.npy'.format(env_name[0], number)))
    reward3 = smooth(np.load('./data_train/PPO_env_{}_number_{}_seed_100.npy'.format(env_name[0], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]
    sns.tsplot(time=np.arange(len), data=reward, color=color, linestyle='-')  # color=darkorange dodgerblue
    plt.plot(reward.mean(0), color=color, label=label)
    plt.title("CartPole-v1", size=14)
    plt.xlabel("Steps", size=14)
    plt.ylabel("Reward", size=14)
    plt.xticks([0, 10, 20, 30, 39], ['0', '50k', '100k', '150k', '200k'], size=14)
    plt.yticks(size=14)
    plt.ylim([0, 510])
    plt.legend(loc='lower right', fontsize=14)


def drawing_LL(plt, number, color, label):
    reward1 = smooth(np.load('./data_train/PPO_env_{}_number_{}_seed_0.npy'.format(env_name[1], number)))
    reward2 = smooth(np.load('./data_train/PPO_env_{}_number_{}_seed_10.npy'.format(env_name[1], number)))
    reward3 = smooth(np.load('./data_train/PPO_env_{}_number_{}_seed_100.npy'.format(env_name[1], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]
    sns.tsplot(time=np.arange(len), data=reward, color=color, linestyle='-')  # color=darkorange dodgerblue
    plt.plot(reward.mean(0), color=color, label=label)
    plt.title("LunarLander-v2", size=14)
    plt.xlabel("Steps", size=14)
    plt.ylabel("Reward", size=14)
    plt.xticks([0, 40, 80, 120, 160, 200], ['0', '200k', '400k', '600k', '800k', '1M'], size=14)
    plt.yticks(size=12)
    plt.ylim([-400, 300])
    plt.legend(loc='lower right', fontsize=14)


sns.set_style('darkgrid')
plt.figure()

plt.subplot(1, 2, 1)
# drawing_CP(plt, number=1, color=colors[1], label='number 1')
# drawing_CP(plt, number=2, color=colors[2], label='number 2')
drawing_CP(plt, number=3, color=colors[1], label='PPO+GRU')
drawing_CP(plt, number=5, color=colors[2], label='PPO+LSTM')

plt.subplot(1, 2, 2)
# drawing_LL(plt, number=1, color=colors[1], label='number 1')
# drawing_LL(plt, number=2, color=colors[2], label='number 2')
drawing_LL(plt, number=3, color=colors[1], label='PPO+GRU')
drawing_LL(plt, number=5, color=colors[2], label='PPO+LSTM')

plt.show()

