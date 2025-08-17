import torch
import torch.nn as nn
from ..builder import BACKBONES
from mmcv.runner import BaseModule
@BACKBONES.register_module()
class FCDiscriminator(BaseModule):
    """
    Fully convolutional discriminator as in Radford et al. (2015)
    Used in the paper for adversarial domain adaptation in output space.
    """
    def __init__(self, num_classes=7):
        super(FCDiscriminator, self).__init__()
        # num_classes is C in output maps (or 1 if you feed entropy maps)
        self.model = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1)  # Output 1 channel for real/fake score
        )

    def forward(self, x):
        """
        x: Tensor of shape (N, C, H, W)
           - For entropy maps: C = 1
           - For segmentation outputs: C = num_classes
        """
        return self.model(x)
