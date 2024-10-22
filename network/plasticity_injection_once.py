import torch.nn as nn
from .base_network import Large_Network

class Plasticity_Injection_Once(Large_Network):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)

        self.plasticity_bias = nn.Linear(512, env.single_action_space.n)

        self.plasticity_bias_correction = nn.Linear(512, env.single_action_space.n)
        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())

        self.plasticity_bias_linear = nn.Linear(3136, 512)

        self.plasticity_bias_correction_linear = nn.Linear(3136, 512)
        self.plasticity_bias_correction_linear.load_state_dict(self.plasticity_bias_linear.state_dict())

        self._change_grad(self.plasticity_bias, False)
        self._change_grad(self.plasticity_bias_correction, False)

        self._change_grad(self.plasticity_bias_linear, False)
        self._change_grad(self.plasticity_bias_correction_linear, False)

    def _forward(self, x):
        x = self.relu(self.conv2d_1(x))
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.flatten(x)
        x = self.linear(x) + (self.plasticity_bias_linear(x) - self.plasticity_bias_correction_linear(x))
        x = self.relu(x)
        return self.head(x) + (self.plasticity_bias(x) - self.plasticity_bias_correction(x))

    def every_step(self, step, num_steps):

        if float(step) / float(num_steps) > 0.5 and not self.plasticity_bias.weight.requires_grad:
            print("Injecting plasticity")
            self._change_grad(self.plasticity_bias, True)
            self._change_grad(self.plasticity_bias_linear, True)
            self._change_grad(self.head, False)
    
    def _change_grad(self, layer, requires_grad):
        for param in layer.parameters():
            param.requires_grad = requires_grad
