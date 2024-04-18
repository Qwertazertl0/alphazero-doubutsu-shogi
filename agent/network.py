import torch
import torch.nn as nn
from game_state import TIME_HISTORY

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        out += identity
        out = self.relu(out)
        
        return out

class AgentNetwork(nn.Module):
    def __init__(self, num_blocks, *args, num_filters=256, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Main network
        self.network = nn.Sequential(
            nn.Conv2d(18 * TIME_HISTORY + 2, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            *[ResBlock(num_filters, num_filters) for _ in range(num_blocks)]
        )

        # Output policy of shape 12 x 4 x 3
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 24, kernel_size=1, stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24*4*3, 12*4*3)
        )

        # Output scalar value
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12, 1),
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.network(x)
        policy = self.policy_head(latent)
        policy = torch.reshape(policy, (12, 4, 3))
        value = self.value_head(latent)
        return policy, value