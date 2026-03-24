import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Determine output size of conv layers
        with torch.no_grad():
            # dummy = torch.rand(1, 1, 84, 84)
            # conv_out_size = self.conv(dummy).shape[1]
            dummy = torch.rand(1, *input_shape)#.permute(0, 3, 1, 2)
            conv_out_size = self.conv(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.Tanh(),
            nn.Linear(512, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # Expecting input as [B, 1, 84, 84]
        features = self.conv(x)
        return Categorical(logits=self.actor(features)), self.critic(features)
