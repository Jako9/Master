from network.base_network import Plastic
from torch import nn

class Large_DNN(Plastic):
    """
    Standard large deep neural network with 3 convolutional layers and 2 linear layers.
    It provides interfaces for the three common plasticity methods.

    1) Use external modifications: Don't alter anything in the network, just use the external modifications.
    2) Modify the networks head: Use self.body and self.head to easily work on the output layer alone.
    3) Modify the whole network: Use self.layer__name for each layer to work on the whole network.
    """
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv2d_1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2d_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(3136, 512)
        self.head = nn.Linear(512, env.single_action_space.n)

        self.body = nn.Sequential(
            self.conv2d_1,
            self.relu,
            self.conv2d_2,
            self.relu,
            self.conv2d_3,
            self.relu,
            self.flatten,
            self.linear,
            self.relu
        )

    def _forward(self, x, global_step):
        x = self.body(x)
        return self.head(x)