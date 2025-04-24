import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from Replaybuffer import ReplayBuffer

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        Q = V + A - T.mean(A, dim=-1, keepdim=True)
        return Q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))

class ST_D3QN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma, tau, epsilon, eps_end, max_size, batch_size):
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilon
        self.epsilon_min = eps_end
        self.episode_start = epsilon
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.max_step = 1.0e4

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        self.memory = ReplayBuffer(max_size)

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, eval_param in zip(self.q_target.parameters(), self.q_eval.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, state_, done):
        self.memory.push(state, action, reward, state_, done)

    def decrement_epsilon(self, step):
        self.epsilon = self.epsilon_min + (self.episode_start - self.epsilon_min) * \
                       math.exp(-1.0 * step / self.max_step)
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def choose_action(self, observation, action_space, isTrain=True):
        if (np.random.random() > self.epsilon) and isTrain:
            state = T.tensor([observation], dtype=T.float).to(device)
            q_vals = self.q_eval.forward(state)
            action = T.argmax(q_vals).item()
        else:
            action = np.random.choice(action_space)
        return action

    def learn(self):
        if not self.memory.ready(self.batch_size):
            return

        states, actions, rewards, next_states, terminals = self.memory.sample(self.batch_size)

        batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)

        with T.no_grad():
            q_next = self.q_target.forward(next_states_tensor)
            max_actions = T.argmax(self.q_eval.forward(next_states_tensor), dim=-1)
            q_next[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_next[batch_idx, max_actions]

        q_eval = self.q_eval.forward(states_tensor)[batch_idx, actions_tensor]

        loss = F.mse_loss(q_eval, target)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()

        return loss.item()

    def save_models(self):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/ST_D3QN_q_eval.pth')
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/ST_D3QN_Q_target.pth')
        print('Saving Q_target network successfully!')

    def load_models(self):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/ST_D3QN_q_eval.pth')
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/ST_D3QN_Q_target.pth')
        print('Loading Q_target network successfully!')