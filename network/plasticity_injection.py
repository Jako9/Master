import torch.nn as nn
from .base_network import Large_Network

class Plasticity_Injection(Large_Network):
    def __init__(self, env, *args, **kwargs):
        super().__init__()

        self.plasticity_bias = nn.Linear(512, env.single_action_space.n)

        self.plasticity_bias_correction = nn.Linear(512, env.single_action_space.n)
        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())

        self._change_grad(self.head, False)
        self._change_grad(self.plasticity_bias_correction, False)

    def forward(self, x):
        x = self.body(x / 255.0)
        return self.head(x) + (self.plasticity_bias(x) - self.plasticity_bias_correction(x))

    def every_drift(self, num_drift):

        self.head.weight = nn.Parameter(self.head.weight + self.plasticity_bias.weight - self.plasticity_bias_correction.weight)
        self.head.bias = nn.Parameter(self.head.bias + self.plasticity_bias.bias - self.plasticity_bias_correction.bias)


        self.plasticity_bias.reset_parameters()
        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())
    
    def _change_grad(self, layer, requires_grad):
        for param in layer.parameters():
            param.requires_grad = requires_grad
