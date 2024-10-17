import torch.nn as nn
from .base_network import Plastic

class Plasticity_Injection(Plastic):
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

        self.plasticity_bias = nn.Linear(512, env.single_action_space.n)

        self.plasticity_bias_correction = nn.Linear(512, env.single_action_space.n)
        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())

        self._change_grad(self.head, False)
        self._change_grad(self.plasticity_bias_correction, False)

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.head(x) + (self.plasticity_bias(x) - self.plasticity_bias_correction(x))

    def every_drift(self, num_drift):

        self.head.weight = nn.Parameter(self.head.weight + self.plasticity_bias.weight - self.plasticity_bias_correction.weight)
        self.head.bias = nn.Parameter(self.head.bias + self.plasticity_bias.bias - self.plasticity_bias_correction.bias)

        self.plasticity_bias = nn.Linear(512, self.head.out_features).to(self.head.weight.device)

        self.plasticity_bias_correction = nn.Linear(512, self.head.out_features).to(self.head.weight.device)
        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())

        self._change_grad(self.plasticity_bias, True)
        self._change_grad(self.plasticity_bias_correction, False)
        self._change_grad(self.head, False)
    
    def _change_grad(self, layer, requires_grad):
        for param in layer.parameters():
            param.requires_grad = requires_grad
