import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.data = []

    def add_trajectory(self, trajectory):
        for transition in trajectory:
            if len(self.data) >= self.capacity:
                self.data.pop(0)
            self.data.append(transition)

    def sample(self, batch_size):
        return random.sample(self.data, min(batch_size, len(self.data)))

    def __len__(self):
        return len(self.data)
