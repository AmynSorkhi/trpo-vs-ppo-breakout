import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """Initialize the policy network.

        Args:
            input_shape: Shape of input observations (H, W, C)
            num_actions: Number of possible actions
        """
        super().__init__()

        # Shared convolutional layers for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate convolutional output size dynamically
        with torch.no_grad():
            # Create dummy input to determine feature size
            dummy = torch.rand(1, *input_shape).permute(0, 3, 1, 2)  # NHWC to NCHW
            conv_out = self.conv(dummy).shape[1]

        # Actor network - outputs action distribution parameters
        self.actor = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.Tanh(),
            nn.Linear(512, num_actions)
        )

        # Critic network - outputs state value estimate
        self.critic = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def actor_forward(self, x):
        """Forward pass for actor network.

        Args:
            x: Input state tensor (batch_size, H, W, C)

        Returns:
            Categorical distribution over actions
        """
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        features = self.conv(x)
        return torch.distributions.Categorical(logits=self.actor(features))

    def critic_forward(self, x):
        """Forward pass for critic network.

        Args:
            x: Input state tensor (batch_size, H, W, C)

        Returns:
            State value estimate tensor (batch_size, 1)
        """
        x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
        features = self.conv(x)
        return self.critic(features)