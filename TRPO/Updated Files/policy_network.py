import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    """Policy network with actor-critic architecture for TRPO.
     Based on:
    - Mnih et al. (2015) - Original Atari CNN architecture
    - Schulman et al. (2015) - TRPO methodology
    - OpenAI Baselines (2017) - Implementation details

    The network consists of:
    - A shared convolutional feature extractor
    - An actor head that outputs action distributions
    - A critic head that outputs state values
    
    """
    def __init__(self, input_shape, num_actions):
        super().__init__()
        # If input_shape has 4 dimensions (e.g., (84,84,1,4)), merge the last two dims.
        if len(input_shape) == 4:
            input_shape = input_shape[:2] + (input_shape[2] * input_shape[3],)
            print("len 4 if")
        # Now input_shape should be (H, W, C)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.rand(1, *input_shape).permute(0, 3, 1, 2)  # NHWC -> NCHW
            conv_out = self.conv(dummy).shape[1]
        self.actor = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(), #CHANGED FROM TANH
            nn.Linear(512, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        self.actor.apply(self._init_weights)
        self.critic.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)


    def actor_forward(self, x):
        # x is expected to be NHWC; convert to NCHW.
        x = x.permute(0, 3, 1, 2)
        features = self.conv(x)
        return torch.distributions.Categorical(logits=self.actor(features))

    def critic_forward(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.conv(x)
        return self.critic(features)
