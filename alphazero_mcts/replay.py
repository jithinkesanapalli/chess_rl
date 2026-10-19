"""Simple bounded replay buffer with atomic-ish pickle persistence."""
import os, pickle, random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.data = deque(maxlen=capacity)
    def __len__(self):
        return len(self.data)
    def add_many(self, samples):
        self.data.extend(samples)
    def sample(self, n):
        return random.sample(list(self.data), min(n, len(self.data)))
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(list(self.data), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.data.extend(pickle.load(f))
