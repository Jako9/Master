import torch
import torch.nn as nn

class QNetwork(nn.Module):
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
        self.head_ref = nn.Linear(512, env.single_action_space.n)
        self.head_ref.load_state_dict(self.head.state_dict())
        self.plasticity_bias = nn.Linear(512, env.single_action_space.n)
        self.plasticity_bias_correction = nn.Linear(512, env.single_action_space.n)
        self.plasticity_bias_correction.load_state_dict(self.plasticity_bias.state_dict())
        
        self.injected = False

        self._change_grad(self.head_ref, False)
        self._change_grad(self.plasticity_bias, False)
        self._change_grad(self.plasticity_bias_correction, False)

    def forward(self, x):
        x = self.network(x / 255.0)

        #Before plasticity injection
        if not self.injected:
            return self.head(x)
        
        #After plasticity injection
        return self.head(x) + (self.plasticity_bias(x) - self.plasticity_bias_correction(x))
    
    def inject_plasticity(self):
        #Update head comparison only for sanity check
        self.head_ref.load_state_dict(self.head.state_dict())

        self._change_grad(self.head, False)
        self._change_grad(self.head_ref, False)
        self._change_grad(self.plasticity_bias, True)

        self.injected = True

    def check_plasticity_status(self):
        return (torch.equal(self.head.weight, self.head_ref.weight) ^ (not self.injected)) and (torch.equal(self.plasticity_bias.weight, self.plasticity_bias_correction.weight) ^ self.injected)    
    
    def _change_grad(self, layer, requires_grad):
        for param in layer.parameters():
            param.requires_grad = requires_grad
