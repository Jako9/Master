import torch.nn as nn
from abc import ABC, abstractmethod
import wandb

"""
Abstract class for all networks.
This class provides an interface for hooking on different levels on the training loop.

To implement the fitting plasticity method just override the method corresponding to the correct hook-level.
Also implement the _forward method which is the forward pass of the network.

1) every_init: This method is called once at the beginning of the training loop.
2) every_drift: This method is called after every drift.
3) every_step: This method is called after every step in the environment.
"""
class Plastic(ABC, nn.Module):

    def __init__(self, total_steps, total_drifts, track, *args, **kwargs):
        super().__init__()
        self.track = track
        self.total_steps = total_steps
        self.total_drifts = total_drifts

    def forward(self, x, global_step=None):
        return self._forward(x / 255.0, global_step)
    
    @abstractmethod
    def _forward(self, x, global_step):
        raise NotImplementedError

    def init_params(self, num_drifts, num_steps):
        self.total_steps = num_steps
        self.total_drifts = num_drifts
    
    def every_init(self):
        pass

    def every_drift(self, num_drift):
        pass

    def every_step(self, step):
        pass

"""
Standard large deep neural network with 3 convolutional layers and 2 linear layers.
It provides interfaces for the three common plasticity methods.

1) Use external modifications: Don't alter anything in the network, just use the external modifications.
2) Modify the networks head: Use self.body and self.head to easily work on the output layer alone.
3) Modify the whole network: Use self.layer__name for each layer to work on the whole network.
"""
class Large_DNN(Plastic):
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
    
class Small_DNN(Plastic):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv2d_1 = nn.Conv2d(4, 64, 8, stride=2)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(97344, 512)
        self.head = nn.Linear(512, env.single_action_space.n)

        self.body = nn.Sequential(
            self.conv2d_1,
            self.relu,
            self.flatten,
            self.linear,
            self.relu
        )

    def _forward(self, x, global_step):
        x = self.body(x)
        return self.head(x)
    
import snntorch as snn
import torch
import numpy as np
from snntorch import spikegen
from utils import add_log

"""
Standard large deep spiking neural network with 3 convolutional layers and 2 linear layers.
It provides interfaces for the three common plasticity methods.

1) Use external modifications: Don't alter anything in the network, just use the external modifications.
2) Modify the networks head: Use self.body and self.head to easily work on the output layer alone.
3) Modify the whole network: Use self.layer__name for each layer to work on the whole network.
"""
class Large_SNN(Plastic):
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

import torch.nn.init as init
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR100 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(Plastic):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(4, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.linear = nn.Linear(64, env.single_action_space.n)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward(self, x, global_step):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResNet18(Plastic):
    from torchvision.models import resnet18
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, env.single_action_space.n)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=1, bias=False)

    def _forward(self, x, global_step):
        return self.model(x)
    
"""from spikingjelly.clock_driven import layer
from spikingjelly.cext import neuron as cext_neuron
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock_SNN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(BasicBlock_SNN, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = cext_neuron.MultiStepIFNode(detach_reset=True)

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = cext_neuron.MultiStepIFNode(detach_reset=True)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))

        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        nn.init.constant_(m.conv2.module[1].weight, 0)
        if connect_f == 'AND':
            nn.init.constant_(m.conv2.module[1].bias, 1)


class ResNet18SNN(Plastic):

    def __init__(self, env, *args, **kwargs):
        super(ResNet18SNN, self).__init__()
        zero_init_residual=False
        replace_stride_with_dilation=None
        norm_layer=None
        layers = [2, 2, 2, 2]
        self.T = 4
        self.connect_f = "ADD"
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)


        self.sn1 = cext_neuron.MultiStepIFNode(detach_reset=True)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(BasicBlock_SNN, 64, layers[0], connect_f=self.connect_f)
        self.layer2 = self._make_layer(BasicBlock_SNN, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=self.connect_f)
        self.layer3 = self._make_layer(BasicBlock_SNN, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=self.connect_f)
        self.layer4 = self._make_layer(BasicBlock_SNN, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=self.connect_f)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * BasicBlock_SNN.expansion, env.single_action_space.n)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, self.connect_f)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                cext_neuron.MultiStepIFNode(detach_reset=True)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        return self.fc(x.mean(dim=0))

    def forward(self, x):
        return self._forward_impl(x)"""

