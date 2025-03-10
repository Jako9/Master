import snntorch as snn
import torch
from snntorch import spikegen
from utils import add_log
from torch import nn
from network.base_network import Plastic


class Large_SNN(Plastic):
    """
    Standard large deep spiking neural network with 3 convolutional layers and 2 linear layers.
    It provides interfaces for the three common plasticity methods.

    1) Use external modifications: Don't alter anything in the network, just use the external modifications.
    2) Modify the networks head: Use self.body and self.head to easily work on the output layer alone.
    3) Modify the whole network: Use self.layer__name for each layer to work on the whole network.
    """
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_steps = 10

        self.action_space = env.single_action_space.n

        self.conv2d_1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2d_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(3136, 512)
        self.head = nn.Linear(512, self.action_space)

        self.pop1 = PopNorm([32, 20, 20], threshold=1, v_reset=0)
        self.pop2 = PopNorm([64, 9, 9], threshold=1, v_reset=0)
        self.pop3 = PopNorm([64, 7, 7], threshold=1, v_reset=0)
        self.pop_fc = PopNorm(512, threshold=1, v_reset=0)

        self.lif1 = snn.Leaky(beta=0.95)
        self.lif2 = snn.Leaky(beta=0.95)
        self.lif3 = snn.Leaky(beta=0.95)
        self.lif_fc = snn.Leaky(beta=0.95)
        #TODO: Implement a real LI head instead of modifying LIF
        #self.lif_head = snn.Leaky(beta=0.95, threshold=np.iinfo(np.int32).max)
        self.head = nn.Linear(512, self.action_space)

        self.body = nn.Sequential(
            self.conv2d_1,
            self.lif1,
            self.flatten,
            self.linear,
            self.lif_fc
        )

    def _forward(self, x, global_step):

        size = x.size(0)
        """x = x.unsqueeze(0).expand(10, -1, -1, -1, -1)
        random_values = torch.rand_like(x).to(x.device)
        spike_train = (random_values < x).float().to(x.device)"""

        spike_train = spikegen.rate(x, num_steps=self.num_steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_fc = self.lif_fc.init_leaky()
        #mem_head = self.lif_head.init_leaky()

        output = torch.zeros(size, self.action_space).to(x.device)

        spk1_average = 0
        spk2_average = 0
        spk3_average = 0
        spk_fc_average = 0
        
        for step in range(self.num_steps):
            out = self.conv2d_1(spike_train[step])
            out = self.pop1(out)
            spk1, mem1 = self.lif1(out, mem1)

            out = self.conv2d_2(spk1)
            out = self.pop2(out)
            spk2, mem2 = self.lif2(out, mem2)

            out = self.conv2d_3(spk2)
            out = self.pop3(out)
            spk3, mem3 = self.lif3(out, mem3)

            out = self.flatten(spk3)
            out = self.linear(out)
            out = self.pop_fc(out)

            spk_fc, mem_fc = self.lif_fc(out, mem_fc)

            output += (self.head(spk_fc) / self.num_steps)

            #_, mem_head = self.lif_head(out, mem_head)

            #Only for logging purposes
            if self.track and global_step is not None and global_step % 100 == 0:
                spk1_average += spk1.mean().item()
                spk2_average += spk2.mean().item()
                spk3_average += spk3.mean().item()
                spk_fc_average += spk_fc.mean().item()

            #TODO: Be able to add other pooling methods (mean, last, etc.)
            #mem_out = torch.max(mem_head, mem_out)
        
        #out = self.head(spk_fc)
        if self.track and global_step is not None and global_step % 100 == 0:
            add_log("spikes/layer1", spk1_average / self.num_steps)
            add_log("spikes/layer2", spk2_average / self.num_steps)
            add_log("spikes/layer3", spk3_average / self.num_steps)
            add_log("spikes/fc", spk_fc_average / self.num_steps)

        return output



from torch.nn import Module, Parameter, functional as F
from torch import Tensor, Size
from typing import List, Union
import numbers
_shape_t = Union[int, List[int], Size]
class PopNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: _shape_t
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, threshold: float, v_reset: float, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.threshold = threshold
        self.v_reset = v_reset
        self.eps = eps
        self.affine = affine
        if self.affine:
            # self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            # nn.init.ones_(self.weight)
            # nn.init.zeros_(self.bias)
            nn.init.constant_(self.weight, self.threshold-self.v_reset)
            nn.init.constant_(self.bias, self.v_reset)
    def forward(self, input: Tensor) -> Tensor:
        out = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)
        # out = F.layer_norm(
        #     input, self.normalized_shape, None, None, self.eps)
        # if self.affine:
        #     out = self.weight * out + self.bias
        return out
    def extra_repr(self) -> Tensor:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)