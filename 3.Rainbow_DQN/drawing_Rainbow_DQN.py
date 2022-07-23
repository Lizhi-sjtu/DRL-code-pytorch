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
colors = ['r', 'darkorange', 'dodgerblue', 'limegreen', 'yellow', 'magenta', 'chocolate', 'indigo', 'gray', 'aqua', 'g', 'black']


def get_data(algorithm, env_index, number):
    reward1 = smooth(np.load('./data_train/{}_env_{}_number_{}_seed_0.npy'.format(algorithm, env_name[env_index], number)))
    reward2 = smooth(np.load('./data_train/{}_env_{}_number_{}_seed_10.npy'.format(algorithm, env_name[env_index], number)))
    reward3 = smooth(np.load('./data_train/{}_env_{}_number_{}_seed_100.npy'.format(algorithm, env_name[env_index], number)))
    reward = np.stack((reward1, reward2, reward3), axis=0)
    len = reward1.shape[0]

    return reward, len


def drawing_CP(plt, algorithm, number, color, label):
    reward, len = get_data(algorithm=algorithm, env_index=0, number=number)
    sns.tsplot(time=np.arange(len), data=reward, color=color, linestyle='-')  # color=darkorange dodgerblue
    plt.plot(reward.mean(0), color=color, label=label)
    plt.title("CartPole-v1", size=14)
    plt.xlabel("Steps", size=14)
    plt.ylabel("Reward", size=14)
    plt.xticks([0, 50, 100, 150], ['0', '50k', '100k', '150k'], size=14)
    plt.yticks(size=14)
    plt.ylim([0, 510])
    plt.legend(loc='lower right', fontsize=14)


def drawing_LL(plt, algorithm, number, color, label):
    reward, len = get_data(algorithm=algorithm, env_index=1, number=number)
    sns.tsplot(time=np.arange(len), data=reward, color=color, linestyle='-')  # color=darkorange dodgerblue
    plt.plot(reward.mean(0), color=color, label=label)
    plt.title("LunarLander-v2", size=14)
    plt.xlabel("Steps", size=14)
    plt.ylabel("Reward", size=14)
    plt.xticks([0, 100, 200, 300, 400], ['0', '100k', '200k', '300k', '400k'], size=14)
    plt.yticks(size=14)
    plt.ylim([-300, 300])
    plt.legend(loc='lower right', fontsize=14)


sns.set_style('darkgrid')
plt.figure()
drawing_LL(plt, algorithm='Rainbow_DQN', number=1, color=colors[0], label='Rainbow_DQN')

drawing_LL(plt, algorithm='DQN_dueling_Noisy_PER_N_steps', number=1, color=colors[1], label='Rainbow_DQN without Double')

drawing_LL(plt, algorithm='DQN_double_Noisy_PER_N_steps', number=1, color=colors[2], label='Rainbow_DQN without Dueling')

drawing_LL(plt, algorithm='DQN_double_dueling_Noisy_N_steps', number=1, color=colors[3], label='Rainbow DQN without PER')

drawing_LL(plt, algorithm='DQN_double_dueling_Noisy_PER', number=1, color=colors[4], label='Rainbow_DQN without N-steps')

drawing_LL(plt, algorithm='DQN_double_dueling_PER_N_steps', number=1, color=colors[9], label='Rainbow_DQN without Noisy')


plt.show()
