import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, obs_space, n_actions):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_space, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, n_actions)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    def forward(self, obs):
        x = obs
        x = fn.relu(self.linear1(x))
        x = fn.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    def reset(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    def set_params(self, state_dict):
        self.load_state_dict(state_dict)

class DeepQAgent:
    def __init__(self, obs_space, n_actions, lr=0.001, learn_update_rule="SGD", batch_size=512, update_frequency=1000
                 ,replay_buffer_size=1000, eps_init=1, eps_min=0.01, eps_decay=1e-6, gamma=0.9):
        self.action_space = [i for i in range(n_actions)]
        self.q_eval = QNetwork(obs_space=obs_space, n_actions=n_actions)
        self.q_target = QNetwork(obs_space=obs_space, n_actions=n_actions)
        self.q_target.set_params(self.q_eval.state_dict())
        self.lr = lr
        if learn_update_rule == "SGD":
            self.optimizer = optim.SGD(self.q_eval.parameters(), lr=lr)
        elif learn_update_rule == "Adam":
            self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)
        else:
            raise ValueError("Invalid learn update rule")
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.replay_buffer_size = replay_buffer_size
        self.eps_init = eps_init
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma

        self.obs_memory = np.zeros((self.replay_buffer_size, obs_space), dtype=np.float32)
        self.next_obs_memory = np.zeros((self.replay_buffer_size, obs_space), dtype=np.float32)
        self.action_memory = np.zeros(self.replay_buffer_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.replay_buffer_size, dtype=np.float32)
        self.termination_memory = np.zeros(self.replay_buffer_size, dtype=np.bool)

        self.mem_cntr = 0
        self.iter_cntr = 0
    def choose_action(self, obs):
        if np.random.random() < self.eps:
            return np.random.choice(self.action_space)
        else:
            obs = torch.tensor(np.array([obs]), dtype=torch.float32)
            action_values = self.q_target(obs)
            return torch.argmax(action_values).item()
    def choose_greedy_action(self, obs):
        obs = torch.tensor(np.array([obs]), dtype=torch.float32)
        action_values = self.q_target(obs)
        return torch.argmax(action_values).item()
    def reset(self):
        self.q_eval.reset()
        self.q_target.reset()
        self.eps = self.eps_init
    def load_policy(self, state_dict):
        self.q_eval.set_params(state_dict)
        self.q_target.set_params(state_dict)
    def store_experience(self, obs, action, reward, next_obs, done):
        idx = self.mem_cntr % self.replay_buffer_size
        self.obs_memory[idx] = obs
        self.next_obs_memory[idx] = next_obs
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.termination_memory[idx] = done

        self.mem_cntr += 1
    def train(self):
        if self.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.replay_buffer_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_indices = np.arange(self.batch_size, dtype=np.int32)

        obs_batch = torch.tensor(self.obs_memory[batch]).to(self.q_eval.device)
        next_obs_batch = torch.tensor(self.next_obs_memory[batch]).to(self.q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        termination_batch = torch.tensor(self.termination_memory[batch]).to(self.q_eval.device)

        q_evaluated = self.q_eval(obs_batch)[batch_indices, action_batch]
        q_next = self.q_target(next_obs_batch)
        q_next[termination_batch] = 0

        target_value = reward_batch + self.gamma*torch.max(q_next, dim=1)[0]

        criterion = nn.MSELoss()
        loss = criterion(target_value, q_evaluated).to(self.q_eval.device)
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.eps_min, self.eps - self.eps_decay)

        if self.iter_cntr % self.update_frequency == 0:
            self.q_target.set_params(self.q_eval.state_dict())

        self.iter_cntr += 1
    def set_params(self, state_dict):
        self.q_eval.set_params(state_dict)
        self.q_target.set_params(state_dict)
    def get_params(self):
        return self.q_target.state_dict()

class PERAgent:
    def __init__(self, obs_space, n_actions, lr=0.001, learn_update_rule="SGD", batch_size=512, update_frequency=1000,
                 replay_buffer_size=1000, eps_init=1, eps_min=0.01, eps_decay=0.6e-6, gamma=0.9, alpha=0.4, beta=0.2,
                 beta_increment=0.8e-7):
        self.action_space = [i for i in range(n_actions)]
        self.q_eval = QNetwork(obs_space=obs_space, n_actions=n_actions)
        self.q_target = QNetwork(obs_space=obs_space, n_actions=n_actions)
        self.q_target.set_params(self.q_eval.state_dict())
        self.lr = lr
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr) if learn_update_rule == "Adam" else optim.SGD(
            self.q_eval.parameters(), lr=lr)
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.replay_buffer_size = replay_buffer_size
        self.eps_init = eps_init
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.obs_memory = np.zeros((self.replay_buffer_size, obs_space), dtype=np.float32)
        self.next_obs_memory = np.zeros((self.replay_buffer_size, obs_space), dtype=np.float32)
        self.action_memory = np.zeros(self.replay_buffer_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.replay_buffer_size, dtype=np.float32)
        self.termination_memory = np.zeros(self.replay_buffer_size, dtype=np.bool)
        self.priority_memory = np.zeros(self.replay_buffer_size, dtype=np.float32)

        self.mem_cntr = 0
        self.iter_cntr = 0

    def choose_action(self, obs):
        if np.random.random() < self.eps:
            return np.random.choice(self.action_space)
        else:
            obs = torch.tensor(np.array([obs]), dtype=torch.float32)
            action_values = self.q_target(obs)
            return torch.argmax(action_values).item()
    def choose_greedy_action(self, obs):
        obs = torch.tensor(np.array([obs]), dtype=torch.float32)
        action_values = self.q_target(obs)
        return torch.argmax(action_values).item()
    def reset(self):
        self.q_eval.reset()
        self.q_target.reset()
        self.eps = self.eps_init
    def load_policy(self, state_dict):
        self.q_eval.set_params(state_dict)
        self.q_target.set_params(state_dict)

    def store_experience(self, obs, action, reward, next_obs, done):
        idx = self.mem_cntr % self.replay_buffer_size
        self.obs_memory[idx] = obs
        self.next_obs_memory[idx] = next_obs
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.termination_memory[idx] = done

        self.priority_memory[idx] = self.priority_memory.max() if self.mem_cntr > 0 else 1.0
        self.mem_cntr += 1

    def sample_experience(self):
        max_mem = min(self.mem_cntr, self.replay_buffer_size)

        priorities = self.priority_memory[:max_mem] ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(max_mem, self.batch_size, p=probabilities)

        weights = (max_mem * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return indices, weights

    def train(self):
        if self.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        indices, weights = self.sample_experience()
        batch_indices = np.arange(self.batch_size, dtype=np.int32)

        obs_batch = torch.tensor(self.obs_memory[indices]).to(self.q_eval.device)
        next_obs_batch = torch.tensor(self.next_obs_memory[indices]).to(self.q_eval.device)
        action_batch = self.action_memory[indices].astype(int)
        reward_batch = torch.tensor(self.reward_memory[indices]).to(self.q_eval.device)
        termination_batch = torch.tensor(self.termination_memory[indices]).to(self.q_eval.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.q_eval.device)

        q_evaluated = self.q_eval(obs_batch)[batch_indices, action_batch]
        q_next = self.q_target(next_obs_batch)
        q_next[termination_batch] = 0
        target_value = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = (weights * (q_evaluated - target_value) ** 2).mean()

        loss.backward()
        self.optimizer.step()

        td_errors = torch.abs(q_evaluated - target_value).detach().cpu().numpy()
        self.priority_memory[indices] = td_errors + 1e-6

        if self.iter_cntr % self.update_frequency == 0:
            self.q_target.set_params(self.q_eval.state_dict())

        self.eps = max(self.eps_min, self.eps - self.eps_decay)
        self.iter_cntr += 1
    def set_params(self, state_dict):
        self.q_eval.set_params(state_dict)
        self.q_target.set_params(state_dict)
    def get_params(self):
        return self.q_target.state_dict()

