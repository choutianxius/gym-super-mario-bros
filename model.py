"""
CNN for the policy network
"""

from torch import nn


# neural network
class MarioNet(nn.Module):
    """
    Mini CNN structure
    input
    -> (conv2d + relu) * 3
    -> flatten
    -> (dense + relu) * 2
    -> output
    """
    def __init__(self, output_dim):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        return self.stack(input)
