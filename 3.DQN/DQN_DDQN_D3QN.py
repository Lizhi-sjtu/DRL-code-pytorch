import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import copy


class Dueling_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Dueling_Net, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.V = nn.Linear(hidden_width, 1)
        self.A = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = torch.relu(self.l1(s))
        s = torch.relu(self.l2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=1, keepdim=True))  # Q(s,a;w)=V(s;w)+A(s,a;w)-mean(A(s,a;w))
        return Q


class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Net, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = torch.relu(self.l1(s))
        s = torch.relu(self.l2(s))
        Q = self.l3(s)
        return Q


class ReplayBuffer(object):
    def __init__(self, state_dim):
        self.max_size = int(1e6)  # maximum capacity of the replay buffer
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, 1))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.long)  # In the discrete action space, the type of 'a' must be 'longtensor'
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class DQN(object):
    def __init__(self, state_dim, action_dim, if_double, if_dueling):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_doube = if_double
        self.if_dueling = if_dueling
        self.hidden_width = 200  # The number of neurons in hidden layers of the neural network
        self.batch_size = 512
        self.lr = 1e-4  # learning rate
        self.GAMMA = 0.99  # discount factor
        self.epsilon = 0.2  # epsilon-greedy
        self.epsilon_decay = 0.99  # Decay the exploration
        self.tau = 0.005  # Softly update the target network

        if self.if_dueling:  # Whether to use the 'dueling network'
            self.eval_net = Dueling_Net(state_dim, action_dim, self.hidden_width)
        else:
            self.eval_net = Net(state_dim, action_dim, self.hidden_width)

        self.target_net = copy.deepcopy(self.eval_net)  # Copy the evaluation network to the target network
        for p in self.target_net.parameters():  # Target network do not need gradient
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if deterministic:
            a = self.eval_net(s).argmax(dim=1).detach().item()
        else:
            # epsilon greedy, generate a random number in the interval [0,1], if it larger than epsilon，then select the action that maximizes Q, otherwise randomly select an action
            if np.random.uniform() > self.epsilon:
                a = self.eval_net(s).argmax(dim=1).detach().item()
            else:
                a = np.random.randint(0, self.action_dim)
        return a

    def learn(self, replay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = replay_buffer.sample(self.batch_size)

        q_eval = self.eval_net(batch_s).gather(1, batch_a)  # shape：(batch_size, 1)
        with torch.no_grad():  # q_target has no gradient
            if self.if_doube:  # Whether to use the 'double Q-network'
                # Use eval_net to determine the action
                a_argmax = self.eval_net(batch_s_).argmax(dim=1, keepdim=True)  # shape：(batch_size, 1)
                # Use target_net to calculate the q_target
                q_target = batch_r + self.GAMMA * (1 - batch_dw) * self.target_net(batch_s_).gather(1, a_argmax)  # shape：(batch_size, 1)
            else:
                q_target = batch_r + self.GAMMA * (1 - batch_dw) * self.target_net(batch_s_).max(dim=1, keepdim=True)[0]  # shape：(batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Softly update target network parameters
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def evaluate_policy(env, agent):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


if __name__ == '__main__':
    env_name = ['CartPole-v1', 'LunarLander-v2']
    env_index = 0
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    number = 4
    # Set random seed
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_epsiode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name[env_index]))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_epsiode_steps={}".format(max_epsiode_steps))

    if_double = True  # whether to use 'double Q-network'
    if_dueling = True  # whether to use the 'dueling network'
    if if_double and if_dueling:
        DQN_name = 'D3QN'
    elif if_double:
        DQN_name = 'DDQN'
    else:
        DQN_name = 'DQN'

    replay_buffer = ReplayBuffer(state_dim)
    agent = DQN(state_dim, action_dim, if_double, if_dueling)
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/DQN/{}_env_{}_number_{}_seed_{}'.format(DQN_name, env_name[env_index], number, seed))

    max_train_steps = 2e5  # Maximum number of training steps
    random_steps = 3e3  # Take the random steps in the beginning for the better exploration
    update_freq = 50  # Take 50 steps,then update the networks 50 times
    evaluate_freq = 1e3  # # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    while total_steps < max_train_steps:
        s = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            if total_steps < random_steps:  # Take the random steps in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s, deterministic=False)
            s_, r, done, _ = env.step(a)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_epsiode_steps:
                if env_index == 1:
                    if r <= -100: r = -10  # good for LunarLander
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, r, s_, dw)  # Store the transition
            s = s_

            # Take 50 steps,then update the networks 50 times
            if total_steps >= random_steps and total_steps % update_freq == 0:
                for _ in range(update_freq):
                    agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                agent.epsilon = np.clip(agent.epsilon * agent.epsilon_decay, 0.01, 1)  # Decay the exploration
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t epsilon：{}".format(evaluate_num, evaluate_reward, agent.epsilon))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(DQN_name, env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1
