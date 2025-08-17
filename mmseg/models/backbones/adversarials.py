import torch
import torch.nn as nn

from ..builder import BACKBONES
from mmcv.runner import BaseModule

@BACKBONES.register_module()
class GlobalDiscriminator(BaseModule):
    """Dg on score maps P (C-channel softmax maps).
    Input: [B, C, H, W]; Output: [B, 1, H, W] logits."""
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.InstanceNorm2d(base_channels*2, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
            nn.InstanceNorm2d(base_channels*4, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels*4, 1, 3, 1, 1)
        )
    def forward(self, P):
        return self.net(P)
    
@BACKBONES.register_module()
class SemanticDiscriminatorCSA(BaseModule):
    """CSA: 1x1 conv MLP to 2C channels (domain x class)."""
    def __init__(self, in_channels, num_classes, hidden=1024):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, 2*num_classes, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, Fmap):
        return self.seq(Fmap)