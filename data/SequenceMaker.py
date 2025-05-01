import torch
from torch.utils.data import Dataset
import numpy as np

class RandomGenerator(Dataset):
    def __init__(self, N, L, val_range, seed, normalize_val):
        self.N, self.L = N, L
        self.vmin, self.vmax = val_range
        self.normalize_val = normalize_val
        np.random.seed(seed)
        X, y = [], []
        for _ in range(N):
            seq = np.random.randint(self.vmin, self.vmax, L)
            X.append(seq)
            y.append(np.sort(seq))
        self.X = torch.tensor(X, dtype=torch.float32) / normalize_val
        self.y = torch.tensor(y, dtype=torch.float32) / normalize_val

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.X[i], self.y[i]
