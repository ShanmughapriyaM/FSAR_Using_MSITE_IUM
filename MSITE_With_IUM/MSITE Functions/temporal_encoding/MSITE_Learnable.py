import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        x: [T, D]
        """
        x = F.relu(self.fc1(x))
        x = x.mean(dim=0)          # temporal pooling
        x = self.fc2(x)
        return x
