import torch.nn as nn
from .base_network import Plastic

class Layer_Norm(Plastic):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.LayerNorm([64, 7, 7]),            
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n)
        )
