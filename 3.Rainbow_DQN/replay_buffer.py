import torch
import numpy as np
from collections import deque
from sum_tree import SumTree


class ReplayBuffer(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.current_size = 0
        self.count = 0
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

    def store_transition(self, state, action, reward, next_state, terminal, done):
        self.buffer['state'][self.count] = state
        self.buffer['action'][self.count] = action
        self.buffer['reward'][self.count] = reward
        self.buffer['next_state'][self.count] = next_state
        self.buffer['terminal'][self.count] = terminal
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)

        return batch, None, None


class N_Steps_ReplayBuffer(object):
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.current_size = 0
        self.count = 0
        self.n_steps = args.n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

    def store_transition(self, state, action, reward, next_state, terminal, done):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.buffer['state'][self.count] = state
            self.buffer['action'][self.count] = action
            self.buffer['reward'][self.count] = n_steps_reward
            self.buffer['next_state'][self.count] = next_state
            self.buffer['terminal'][self.count] = terminal
            self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]
        next_state, terminal = self.n_steps_deque[-1][3:5]
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def sample(self, total_steps):
        index = np.random.randint(0, self.current_size, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)

        return batch, None, None


class Prioritized_ReplayBuffer(object):
    def __init__(self, args):
        self.max_train_steps = args.max_train_steps
        self.alpha = args.alpha
        self.beta_init = args.beta_init
        self.beta = args.beta_init
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.current_size = 0
        self.count = 0
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

    def store_transition(self, state, action, reward, next_state, terminal, done):
        self.buffer['state'][self.count] = state
        self.buffer['action'][self.count] = action
        self.buffer['reward'][self.count] = reward
        self.buffer['next_state'][self.count] = next_state
        self.buffer['terminal'][self.count] = terminal
        # 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
        priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
        self.sum_tree.update(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
        self.count = (self.count + 1) % self.buffer_capacity  # When the 'count' reaches buffer_capacity, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size, batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)  # beta：beta_init->1.0
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.float32)

        return batch, batch_index, IS_weight

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)


class N_Steps_Prioritized_ReplayBuffer(object):
    def __init__(self, args):
        self.max_train_steps = args.max_train_steps
        self.alpha = args.alpha
        self.beta_init = args.beta_init
        self.beta = args.beta_init
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.n_steps = args.n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }
        self.current_size = 0
        self.count = 0

    def store_transition(self, state, action, reward, next_state, terminal, done):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.buffer['state'][self.count] = state
            self.buffer['action'][self.count] = action
            self.buffer['reward'][self.count] = n_steps_reward
            self.buffer['next_state'][self.count] = next_state
            self.buffer['terminal'][self.count] = terminal
            # 如果是buffer中的第一条经验，那么指定priority为1.0；否则对于新存入的经验，指定为当前最大的priority
            priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
            self.sum_tree.update(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
            self.count = (self.count + 1) % self.buffer_capacity  # When 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps):
        batch_index, IS_weight = self.sum_tree.get_batch_index(current_size=self.current_size, batch_size=self.batch_size, beta=self.beta)
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)  # beta：beta_init->1.0
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.float32)

        return batch, batch_index, IS_weight

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]  # 获取deque中第一个transition的s和a
        next_state, terminal = self.n_steps_deque[-1][3:5]  # 获取deque中最后一个transition的s'和terminal
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):  # 逆序计算n_steps_reward
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if d:  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)
