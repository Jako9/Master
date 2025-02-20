from .base_network import Plastic
from .large_dnn.plasticity_injection import Plasticity_Injection
from .large_dnn.reset_head import Reset_Head
from .large_dnn.layer_norm import Layer_Norm
from .large_dnn.ridig import Rigid
from .large_snn.rigid_snn import Rigid_SNN
from .large_dnn.reset_network import Reset_Network
from .large_snn.reset_snn import Reset_SNN
from .large_snn.bucket_snn import Bucket_SNN
from .small_dnn.small_ridig import Small_Rigid
from .small_dnn.small_reset_network import Small_Reset_Network
from .resnet.resnet_ridig import Resnet_Rigid
from .resnet.resnet_reset_network import Resnet_Reset_Network
from .resnet18.resnet18_ridig import Resnet18_Rigid
from .resnet18.resnet18_reset_network import Resnet18_Reset_Network
#from .resnet18SNN.resnet18SNN_ridig import Resnet18SNN_Rigid
#from .resnet18SNN.resnet18SNN_reset_network import Resnet18SNN_Reset_Network

__all__ = ['Plastic', 'Plasticity_Injection', 'Layer_Norm', 'Reset_Head',
           'Rigid', 'Rigid_SNN', 'Reset_Network', 'Reset_SNN', 'Bucket_SNN',
           'Small_Rigid', 'Small_Reset_Network', 'Resnet_Rigid',
           'Resnet_Reset_Network', 'Resnet18_Rigid', 'Resnet18_Reset_Network']#,
           #'Resnet18SNN_Rigid', 'Resnet18SNN_Reset_Network']

