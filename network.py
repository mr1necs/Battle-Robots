import torch
import torch.nn as nn

STATE_SIZE = 10
ACTION_SPACE = 5

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(64, ACTION_SPACE)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x
