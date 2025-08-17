"""
Custom multi-optimizer support for mmsegmentation.
Allows registering and building multiple optimizers (e.g., for G, Dg, Ds).
"""

import copy
import inspect
import torch
from mmcv.utils import build_from_cfg
from mmcv.runner.optimizer import OPTIMIZERS, OPTIMIZER_BUILDERS


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        print(_optim)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


# TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    """
    Build optimizer(s). If cfg is a dict of dicts, build multiple optimizers.
    Example:
        optimizer = dict(
            model=dict(type='Adam', lr=1e-4, weight_decay=0.0005),
            discriminator_g=dict(type='Adam', lr=1e-4, betas=(0.5,0.999)),
            discriminator_s=dict(type='Adam', lr=1e-4, betas=(0.5,0.999))
        )
    """
    if any(isinstance(v, dict) for v in cfg.values()):
        # multiple optimizers case
        optimizers = {}
        for key, sub_cfg in cfg.items():
            optimizer_cfg = copy.deepcopy(sub_cfg)
            constructor_type = optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
            optim_constructor = build_optimizer_constructor(
                dict(
                    type=constructor_type,
                    optimizer_cfg=optimizer_cfg,
                    paramwise_cfg=paramwise_cfg))
            optimizers[key] = optim_constructor(model if key in ['model', "discriminator_g", "discriminator_s"] else getattr(model, key))
        return optimizers
    else:
        # single optimizer
        optimizer_cfg = copy.deepcopy(cfg)
        constructor_type = optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
        paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        optim_constructor = build_optimizer_constructor(
            dict(
                type=constructor_type,
                optimizer_cfg=optimizer_cfg,
                paramwise_cfg=paramwise_cfg))
        optimizer = optim_constructor(model)
        return optimizer
