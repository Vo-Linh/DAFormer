
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.models import UDA
from mmseg.models import BaseSegmentor, build_segmentor, build_loss, build_backbone
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import mmcv

def set_requires_grad(module, flag: bool):
    """Enable/disable gradients for a module."""
    for p in module.parameters():
        p.requires_grad = flag

def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module

@UDA.register_module()
class AssDA(BaseSegmentor):
    """
    Implementation of:
    Adversarial Semi-Supervised Domain Adaptation for Semantic Segmentation
    https://arxiv.org/pdf/2312.07370
    """

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()

        # Segmentation network (G)
        self.model = build_segmentor(deepcopy(cfg['model']))

        # Discriminator network (D)
        # For entropy maps: in_channels=1, conv layers {64,128,256,512,1}
        self.discriminator = build_backbone(deepcopy(cfg['discriminator']))

        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']

        self.lambda_adv = cfg.get('lambda_adv', 0.001)  # adv loss weight for G
        self.role = cfg.get('role', 'LTS')  # 'LTS', 'LTT', 'Baseline'

        # Build GAN loss from cfg (must have type, e.g., 'vanilla')
        self.gan_loss = build_loss(cfg['gan_loss'])

    def get_model(self):
        return get_module(self.model)
    def extract_feat(self, img):
        """Extract features using the inner segmentation model."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode and decode image into a segmentation map."""
        return self.get_model().encode_decode(img, img_metas)

    def simple_test(self, img, img_metas, rescale=True):
        """Simple test without test-time augmentation."""
        return self.get_model().simple_test(img, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with test-time augmentation."""
        return self.get_model().aug_test(imgs, img_metas, rescale=rescale)
    
    @staticmethod
    def compute_entropy(logits):
        """Compute entropy maps and per-image entropy."""
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)
        entropy_map = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # (N,H,W)
        entropy_per_image = torch.mean(entropy_map.view(logits.size(0), -1), dim=1)  # (N,)
        return entropy_map, entropy_per_image

    def _run_d(self, ent_map):
        """Run entropy map through D. Input shape: (N,H,W) -> output: (N,1,h',w')."""
        return self.discriminator(ent_map.unsqueeze(1))

    # ---------------------------
    # Forward for generator step
    # ---------------------------
    def forward_train(self,
                      img, img_metas, gt_semantic_seg,                         # Source labeled
                      target_unlabeled_img, target_unlabeled_img_metas,        # Target unlabeled
                      target_labeled_img, target_labeled_gt_semantic_seg, target_labeled_img_metas,      # Target labeled
                      return_feat=False):

        losses = {}

        # 1) Segmentation loss on source
        src_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=return_feat
        )
        losses.update({f"src_{k}": v for k, v in src_losses.items()})

        # 2) Segmentation loss on labeled target
        tgt_labeled_losses = self.get_model().forward_train(
            target_labeled_img, target_labeled_img_metas, target_labeled_gt_semantic_seg,
            return_feat=return_feat
        )
        losses.update({f"tgt_labeled_{k}": v for k, v in tgt_labeled_losses.items()})

        # 3) Adversarial loss for G (D frozen)
        set_requires_grad(self.discriminator, False)

        tgt_unlabeled_logits = self.get_model().encode_decode(
            target_unlabeled_img, target_unlabeled_img_metas
        )
        ent_tu, _ = self.compute_entropy(tgt_unlabeled_logits)

        ent_tl = None
        if self.role in ('LTS', 'LTT'):
            labeled_logits = self.get_model().encode_decode(
                target_labeled_img, target_labeled_img_metas
            )
            ent_tl, _ = self.compute_entropy(labeled_logits)

        adv_loss_g = self.discriminator_loss(ent_tl, ent_tu, mode=self.role)
        losses['adv_loss'] = self.lambda_adv * adv_loss_g

        return losses

    def discriminator_loss(self, ent_labeled, ent_unlabeled, mode='Baseline'):
        """Generator-side adversarial loss: fool D to think target-like maps are source."""
        targets_to_fool = [ent_unlabeled]
        if mode == 'LTT' and ent_labeled is not None:
            targets_to_fool.append(ent_labeled)

        ent_cat = torch.cat(targets_to_fool, dim=0) if len(targets_to_fool) > 1 else targets_to_fool[0]
        d_out = self._run_d(ent_cat)

        # For generator: wants D to think targets are source
        return self.gan_loss(d_out, target_is_real=True, is_disc=False)

    # ---------------------------
    # Forward for discriminator step
    # ---------------------------
    # @torch.no_grad()
    def _entropy_map(self, logits):
        ent, _ = self.compute_entropy(logits)
        return ent

    def forward_train_d(self,
                        img, img_metas, gt_semantic_seg,
                        target_unlabeled_img, target_unlabeled_img_metas,
                        target_labeled_img, target_labeled_img_metas, target_labeled_gt_semantic_seg):
        """
        Discriminator update step: D unfrozen, G frozen.
        Returns dict with 'loss_d'.
        """
        
        set_requires_grad(self.get_model(), False)
        set_requires_grad(self.discriminator, True)

        # Compute entropy maps without grad
        src_logits = self.get_model().encode_decode(img, img_metas)
        ent_src = self._entropy_map(src_logits)

        tu_logits = self.get_model().encode_decode(target_unlabeled_img, target_unlabeled_img_metas)
        ent_tu = self._entropy_map(tu_logits)

        tl_logits = self.get_model().encode_decode(target_labeled_img, target_labeled_img_metas)
        ent_tl = self._entropy_map(tl_logits)

        # Role logic
        if self.role == 'Baseline':
            ent_pos = ent_src
            ent_neg = ent_tu
        elif self.role == 'LTS':
            ent_pos = ent_tl
            ent_neg = ent_tu
        else:  # LTT
            ent_pos = ent_src
            ent_neg = torch.cat([ent_tu, ent_tl], dim=0)

        # Run through D
        d_out_pos = self._run_d(ent_pos)
        d_out_neg = self._run_d(ent_neg)

        # Real loss
        loss_real = self.gan_loss(d_out_pos, target_is_real=True, is_disc=True)
        # Fake loss
        loss_fake = self.gan_loss(d_out_neg, target_is_real=False, is_disc=True)

        loss_d = 0.5 * (loss_real + loss_fake)
        return {'loss_d': loss_d}
    
    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network."""
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                if 'loss' in _key)

        log_vars_detached = OrderedDict()
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars_detached[loss_name] = loss_value.item()

        return loss, log_vars_detached

    def train_step(self, data_batch, optimizer, **kwargs):
        """Supports single optimizer or dict of optimizers."""
        # Resolve optimizers
        if isinstance(optimizer, dict):
            opt_g = optimizer.get('model', None)
            opt_d = optimizer.get('discriminator', None)
            # Fallbacks if keys differ
            if opt_g is None:
                # first optimizer is used for G
                opt_g = list(optimizer.values())[0]
            if opt_d is None:
                # last optimizer is used for D (if only one exists, it will equal opt_g)
                opt_d = list(optimizer.values())[-1]
        else:
            # Single optimizer case
            opt_g = optimizer
            opt_d = optimizer

        # ---------------- G step ----------------
        self.train()
        # (Make sure G has grads; D frozen but keep graph alive -> NO torch.no_grad here)
        for p in self.get_model().parameters():
            p.requires_grad = True
        for p in self.discriminator.parameters():
            p.requires_grad = False

        losses_g = self.forward_train(**data_batch)
        loss_g, log_vars_g = self._parse_losses(losses_g)

        # Sanity: we MUST backprop the tensor from _parse_losses
        assert isinstance(loss_g, torch.Tensor) and loss_g.requires_grad, \
            f"G loss lost grads: type={type(loss_g)}, requires_grad={getattr(loss_g,'requires_grad',None)}"

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # ---------------- D step ----------------
        # Freeze G, unfreeze D
        for p in self.get_model().parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = True

        losses_d = self.forward_train_d(**data_batch)
        loss_d, log_vars_d = self._parse_losses(losses_d)

        # D loss should also be a tensor (but may not require grad if D accidentally frozen)
        assert isinstance(loss_d, torch.Tensor), "D loss is not a tensor"
        assert any(p.requires_grad for p in self.discriminator.parameters()), "D params are frozen during D step"

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # Merge logs (detached values only)
        log_vars = OrderedDict()
        for k, v in log_vars_g.items():
            log_vars[f'G/{k}'] = v
        for k, v in log_vars_d.items():
            log_vars[f'D/{k}'] = v

        # Batch size for runner stats
        try:
            batch_size = data_batch['img'].data[0].size(0)
        except Exception:
            batch_size = data_batch['img'].size(0) if hasattr(data_batch['img'], 'size') else 1

        # Return a tensor for runner (not used for backward here)
        return dict(loss=loss_g + loss_d, log_vars=log_vars, num_samples=batch_size)
