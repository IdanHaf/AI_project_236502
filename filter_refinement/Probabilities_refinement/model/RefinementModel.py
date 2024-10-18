import torch
import torch.nn as nn


class RefinementModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input size = 120 (1st model) + 9 (language model)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(129, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 120),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

