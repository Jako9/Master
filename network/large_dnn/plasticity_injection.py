import torch.nn as nn
from ..base_network import Large_DNN
import numpy as np

class Plasticity_Injection(Large_DNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.num_actions = env.single_action_space.n
        self._init()

    def _forward(self, x, global_step):
        x = self.relu(self.conv2d_1(x))
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.flatten(x)
        out = self.theta(x)
        for bias, correction in zip(self.bias, self.bias_correction):
            out += bias(x) - correction(x)

        return out

    def every_init(self):
        device = next(self.head.parameters()).device
        for bias, correction in zip(self.bias, self.bias_correction):
            bias.to(device)
            correction.to(device)

    def every_drift(self, drift):
        if self.total_drifts == 1: #Continous environment, no injection on drifts
            return
        
        print("Injecting plasticity")
        self._change_grad(self.plasticity_linear[drift], True)
        self._change_grad(self.plasticity[drift], True)
        self._change_grad(self.head, False)
        self._change_grad(self.linear, False)

    def every_step(self, step):
        if self.total_drifts != 1: #Discrete environment, no injection on steps
            return

        if float(step) / float(self.total_steps) > 0.5 and not self.plasticity[0].weight.requires_grad:
            print("Injecting plasticity")
            self._change_grad(self.plasticity[0], True)
            self._change_grad(self.plasticity_linear[0], True)
            self._change_grad(self.head, False)
            self._change_grad(self.linear, False)

    def _init(self):
        self.plasticity_linear = nn.ModuleList()
        self.plasticity = nn.ModuleList()

        self.bias = nn.ModuleList()
        self.bias_correction = nn.ModuleList()

        for _ in range(self.total_drifts):
            linear = nn.Linear(3136, 512)
            self.plasticity_linear.append(linear)

            bias_correction_linear = nn.Linear(3136, 512)
            bias_correction_linear.load_state_dict(linear.state_dict())

            self._change_grad(linear, False)
            self._change_grad(bias_correction_linear, False)

            out = nn.Linear(512, self.num_actions)
            self.plasticity.append(out)

            bias_correction = nn.Linear(512, self.num_actions)
            bias_correction.load_state_dict(out.state_dict())

            self._change_grad(out, False)
            self._change_grad(bias_correction, False)

            self.bias.append(nn.Sequential(linear, nn.ReLU(), out))
            self.bias_correction.append(nn.Sequential(bias_correction_linear, nn.ReLU(), bias_correction))

        self.theta = nn.Sequential(self.linear, self.relu, self.head)

    
    def _change_grad(self, layer, requires_grad):
        for param in layer.parameters():
            param.requires_grad = requires_grad
