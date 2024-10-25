import torch.nn as nn
from ..base_network import Large_DNN

class Plasticity_Injection_Drift(Large_DNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)

        self.plasticity_bias = nn.Linear(512, env.single_action_space.n)

        self.plasticity_bias_correction = nn.Linear(512, env.single_action_space.n)
        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())

        self.plasticity_bias_linear = nn.Linear(3136, 512)

        self.plasticity_bias_correction_linear = nn.Linear(3136, 512)
        self.plasticity_bias_correction_linear.load_state_dict(self.plasticity_bias_linear.state_dict())

        self._change_grad(self.linear, False)
        self._change_grad(self.plasticity_bias_correction, False)

        self._change_grad(self.head, False)
        self._change_grad(self.plasticity_bias_correction_linear, False)

    def _forward(self, x):
        x = self.relu(self.conv2d_1(x))
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.flatten(x)
        x = self.linear(x) + (self.plasticity_bias_linear(x) - self.plasticity_bias_correction_linear(x))
        x = self.relu(x)
        return self.head(x) + (self.plasticity_bias(x) - self.plasticity_bias_correction(x))

    def every_drift(self, num_drift):

        self.head.weight = nn.Parameter(self.head.weight + self.plasticity_bias.weight - self.plasticity_bias_correction.weight)
        self.head.bias = nn.Parameter(self.head.bias + self.plasticity_bias.bias - self.plasticity_bias_correction.bias)

        self.linear.weight = nn.Parameter(self.linear.weight + self.plasticity_bias_linear.weight - self.plasticity_bias_correction_linear.weight)
        self.linear.bias = nn.Parameter(self.linear.bias + self.plasticity_bias_linear.bias - self.plasticity_bias_correction_linear.bias)


        self.plasticity_bias.reset_parameters()
        self.plasticity_bias_linear.reset_parameters()

        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())
        self.plasticity_bias_correction_linear.load_state_dict(self.plasticity_bias_linear.state_dict())
    
    def _change_grad(self, layer, requires_grad):
        for param in layer.parameters():
            param.requires_grad = requires_grad
