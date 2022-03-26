import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import gym


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.tanh(self.l1(s))
        s = F.tanh(self.l2(s))
        a_prob = F.softmax(self.l3(s), dim=1)
        return a_prob


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        v_s = self.l3(s)
        return v_s


class Memory():
    def __init__(self, batch_size, state_dim):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, 1))
        self.a_prob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, s, a, a_prob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_prob[self.count] = a_prob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In the discrete action space, the type of 'a' must be 'longtensor'
        a_prob = torch.tensor(self.a_prob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_prob, r, s_, dw, done


class PPO():
    def __init__(self, state_dim, action_dim, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.mini_batch_size = 64
        self.hidden_width = 64  # The number of neurons in hidden layers of the neural network
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.GAMMA = 0.99  # discount factor
        self.LAMDA = 0.95  # GAE parameter
        self.lr = 3e-4  # learning rate

        self.actor = Actor(state_dim, action_dim, self.hidden_width)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, self.hidden_width)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        prob_weights = self.actor(s).detach().numpy().flatten()  # probability distribution(numpy)
        if deterministic:  # We use the deterministic policy during the evaluating
            a = np.argmax(prob_weights)  # Select the action with the highest probability
            return a
        else:  # We use the stochastic policy during the training
            a = np.random.choice(range(self.action_dim), p=prob_weights)  # Sample the action according to the probability distribution
            a_prob = prob_weights[a]  # The corresponding probability
            return a, a_prob

    def update(self, memory):
        # 获取数据
        s, a, a_prob, r, s_, dw, done = memory.numpy_to_tensor()
        """
            Calculate the advantage using GAE(General Advantage Estimation)
            'dw=True' menas dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.GAMMA * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.GAMMA * self.LAMDA * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            td_target = adv + vs
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))  # The normalization will improve the performance

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                a_prob_now = self.actor(s[index]).gather(1, a[index])
                ratios = a_prob_now / a_prob[index].detach()

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_prob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv[index]
                actor_loss = -torch.min(surr1, surr2).mean()
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                v_s = self.critic(s[index])
                critic_loss = self.MseLoss(td_target[index], v_s)
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


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
    number = 10
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
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    batch_size = 2048  # When the number of transitions in memory reaches batch_size,then update
    max_train_steps = 5e5  # Maximum number of training steps
    evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    memory = Memory(batch_size, state_dim)
    agent = PPO(state_dim, action_dim, batch_size)
    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))

    while total_steps < max_train_steps:
        s = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_prob = agent.choose_action(s, deterministic=False)  # action and the corresponding probability
            s_, r, done, _ = env.step(a)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                if env_index == 1:
                    if r <= -100: r = -10  # good for LunarLander
                dw = True
            else:
                dw = False

            memory.store(s, a, a_prob, r, s_, dw, done)
            s = s_
            # When the number of transitions in memory reaches batch_size,then update
            if memory.count == batch_size:
                agent.update(memory) 
                memory.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1
