# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .aspp_head import ASPPHead
from .da_head import DAHead
from .daformer_head import DAFormerHead, DAFormerVIBHead
from .dlv2_head import DLV2Head
from .fcn_head import FCNHead
from .isa_head import ISAHead
from .psp_head import PSPHead
from .segformer_head import SegFormerHead, SegFormerVIBHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .uper_head import UPerHead
from .vib_head import FPN_VIB_Head
__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'DAHead',
    'DLV2Head',
    'SegFormerHead',
    'SegFormerVIBHead',
    'DAFormerHead',
    'ISAHead',
    'DAFormerVIBHead',
    'FPN_VIB_Head',
]
