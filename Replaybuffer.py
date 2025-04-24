import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminals = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, terminals

    def ready(self, batch_size):
        return len(self.memory) >= batch_size
