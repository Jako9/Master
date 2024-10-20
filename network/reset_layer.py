import torch.nn as nn
from .base_network import Plastic

class Reset_Layer(Plastic):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, env.single_action_space.n)

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.head(x)

    def every_drift(self, num_drift):
        self.head.reset_parameters()
