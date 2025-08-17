from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.models import UDA
from mmseg.models import BaseSegmentor, build_segmentor, build_loss, build_backbone
from mmcv.runner import auto_fp16


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def get_module(module):
    # unwrap DistributedDataParallel if necessary
    try:
        from mmcv.parallel import MMDistributedDataParallel
    except Exception:
        MMDistributedDataParallel = None
    if MMDistributedDataParallel and isinstance(module, MMDistributedDataParallel):
        return module.module
    return module


@UDA.register_module()
class ASSUDA(BaseSegmentor):
    """Adversarial Semi-Supervised Domain Adaptation (GA + SA-CSA)

    cfg expected keys (example):
      model: segmentor config dict
      discriminator_g: backbone config for Dg (in_channels=num_classes)
      discriminator_s: backbone/config for Ds (CSA head expected to output 2*num_classes channels)
      gan_loss: loss cfg (type='GANLoss' or similar) built via build_loss
      lambda_seg, lambda_gadv, lambda_sadv, lambda_gd, lambda_sd
      ignore_index
    """

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()

        self.model = build_segmentor(deepcopy(cfg['model']))

        # Discriminators (Dg, Ds)
        print(cfg['discriminator_g'])
        self.discriminator_g = build_backbone(deepcopy(cfg['discriminator_g']))
        self.discriminator_s = build_backbone(deepcopy(cfg['discriminator_s']))

        self.train_cfg = cfg['model'].get('train_cfg', None)
        self.test_cfg = cfg['model'].get('test_cfg', None)
        self.num_classes = cfg['model']['decode_head']['num_classes']

        # losses and hyperparams
        self.lambda_seg = cfg.get('lambda_seg', 1.0)
        self.lambda_gadv = cfg.get('lambda_gadv', 0.01)
        self.lambda_sadv = cfg.get('lambda_sadv', 0.05)
        self.lambda_gd = cfg.get('lambda_gd', 1.0)
        self.lambda_sd = cfg.get('lambda_sd', 1.0)

        self.ignore_index = cfg.get('ignore_index', 255)

        # build gan loss (should support .forward(input, target_is_real, is_disc))
        self.gan_loss = build_loss(deepcopy(cfg['gan_loss']))

        # standard segmentation loss builder exists in base segmentor; we still use model's forward_train

    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img):
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas, **kwargs):
        return self.get_model().encode_decode(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, rescale=True):
        return self.get_model().simple_test(img, img_metas, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        return self.get_model().aug_test(imgs, img_metas, rescale)

    # ---------------- G step: forward_train ----------------
    def forward_train(self,
                      img, img_metas, gt_semantic_seg,
                      target_unlabeled_img, target_unlabeled_img_metas,
                      target_labeled_img, target_labeled_gt_semantic_seg, target_labeled_img_metas,
                      return_feat=False):
        """Generator step: segmentation supervised losses + adversarial generator losses"""
        losses = {}

        # 1) source supervised: use the inner segmentor forward_train
        src_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=return_feat)
        losses.update({f'src_{k}': v for k, v in src_losses.items()})

        # 2) target labeled supervised
        tgt_labeled_losses = self.get_model().forward_train(target_labeled_img, target_labeled_img_metas,
                                                            target_labeled_gt_semantic_seg, return_feat=return_feat)
        losses.update({f'tgt_l_{k}': v for k, v in tgt_labeled_losses.items()})

        # 3) adversarial generator losses (freeze discriminators)
        set_requires_grad(self.discriminator_g, False)
        set_requires_grad(self.discriminator_s, False)

        # a) global: use score maps (softmax of logits)
        logits_s = self.get_model().encode_decode(img, img_metas)
        logits_tu = self.get_model().encode_decode(target_unlabeled_img, target_unlabeled_img_metas)

        P_s = F.softmax(logits_s, dim=1)
        P_tu = F.softmax(logits_tu, dim=1)

        # run through Dg (assumes Dg accepts C-channel score maps)
        d_out_s = self.discriminator_g(P_s)
        # For G: want Dg(P_s) -> target_is_real True (i.e., make source appear target-like)
        loss_g_gadv = self.gan_loss(d_out_s, target_is_real=True, is_disc=False)
        losses['loss_g_gadv'] = self.lambda_gadv * loss_g_gadv

        # b) semantic CSA: operate on deepest feature maps + labels
        # extract deepest features from feature extractor
        feats_s = self.get_model().extract_feat(img)[-1]
        feats_tl = self.get_model().extract_feat(target_labeled_img)[-1]

        # Ds expected to output [B, 2*C, Hf, Wf]
        logits_ds_s = self.discriminator_s(feats_s)  # [B, 2C, H, W]

        # resize labels to feature size
        Hf, Wf = logits_ds_s.shape[-2], logits_ds_s.shape[-1]
        y_s_ds = F.interpolate(gt_semantic_seg.float(), size=(Hf, Wf), mode='nearest').long().squeeze(1)

        # pick target-half logits (k + C) per pixel and try to make them "real" (True)
        idx_t = (y_s_ds + self.num_classes).unsqueeze(1)  # [B,1,Hf,Wf]
        mask = (y_s_ds != self.ignore_index)

        # clamp idx_t just to be safe
        idx_t = idx_t.clamp(max=2 * self.num_classes - 1)

        # gather only valid pixels
        pick_t = logits_ds_s.gather(1, idx_t).squeeze(1)
        loss_g_sadv = F.binary_cross_entropy_with_logits(
            pick_t[mask], torch.ones_like(pick_t[mask])
        )
        losses['loss_g_sadv'] = self.lambda_sadv * loss_g_sadv

        return losses

    # ---------------- D step: forward_train_d ----------------
    def forward_train_d(self,
                        img, img_metas, gt_semantic_seg,
                        target_unlabeled_img, target_unlabeled_img_metas,
                        target_labeled_img, target_labeled_img_metas, target_labeled_gt_semantic_seg):
        """Discriminator step: update Dg and Ds. G frozen."""
        losses = {}

        set_requires_grad(self.get_model(), False)
        set_requires_grad(self.discriminator_g, True)
        set_requires_grad(self.discriminator_s, True)

        # Compute logits / features (no grad required for G outputs here)
        with torch.no_grad():
            logits_s = self.get_model().encode_decode(img, img_metas)
            logits_tu = self.get_model().encode_decode(target_unlabeled_img, target_unlabeled_img_metas)
            logits_tl = self.get_model().encode_decode(target_labeled_img, target_labeled_img_metas)

            feats_s = self.get_model().extract_feat(img)[-1]
            feats_tl = self.get_model().extract_feat(target_labeled_img)[-1]

        # ----- Dg loss: Ps -> real, Ptu -> fake (or vice-versa depending sign convention of your GANLoss)
        P_s = F.softmax(logits_s, dim=1)
        P_tu = F.softmax(logits_tu, dim=1)

        d_out_s = self.discriminator_g(P_s)
        d_out_tu = self.discriminator_g(P_tu)

        # real = True for source, fake = False for target_unlabeled
        loss_d_real = self.gan_loss(d_out_s, target_is_real=True, is_disc=True)
        loss_d_fake = self.gan_loss(d_out_tu, target_is_real=False, is_disc=True)
        loss_d_g = 0.5 * (loss_d_real + loss_d_fake)
        losses['loss_d_g'] = self.lambda_gd * loss_d_g

        # ----- Ds loss (CSA): source (k) as real, target_labeled (k+C) as fake
        logits_ds_s = self.discriminator_s(feats_s)
        logits_ds_tl = self.discriminator_s(feats_tl)

        Hf, Wf = logits_ds_s.shape[-2], logits_ds_s.shape[-1]
        y_s_ds = F.interpolate(gt_semantic_seg.float(), size=(Hf, Wf), mode='nearest').long().squeeze(1)
        y_tl_ds = F.interpolate(target_labeled_gt_semantic_seg.float(), size=(Hf, Wf), mode='nearest').long().squeeze(1)

        # pick per-pixel channel for source labeled (channel = k)
        idx_src = y_s_ds.unsqueeze(1)

        mask_src = (y_s_ds != self.ignore_index)

        idx_src = idx_src.clamp(max=2 * self.num_classes - 1)

        # safe gather
        pick_src = logits_ds_s.gather(1, idx_src).squeeze(1)

        # compute loss only on valid pixels
        loss_ds_src = F.binary_cross_entropy_with_logits(
            pick_src[mask_src], torch.ones_like(pick_src[mask_src])
        )


        # pick per-pixel channel for target labeled (channel = k + C)
        idx_tgt = (y_tl_ds + self.num_classes).unsqueeze(1)
        mask = (y_tl_ds != self.ignore_index)
        idx_tgt = idx_src.clamp(max=2 * self.num_classes - 1)
        pick_tgt = logits_ds_tl.gather(1, idx_tgt).squeeze(1)
        loss_ds_tgt = F.binary_cross_entropy_with_logits(pick_tgt[mask], torch.zeros_like(pick_tgt[mask]))

        loss_d_s = 0.5 * (loss_ds_src + loss_ds_tgt)
        losses['loss_d_s'] = self.lambda_sd * loss_d_s

        return losses

    # ---------------- generic loss parsing ----------------
    @staticmethod
    def _parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        # gather from all ranks
        log_vars_detached = OrderedDict()
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars_detached[loss_name] = loss_value.item()

        return loss, log_vars_detached

    # ---------------- train_step ----------------
    def train_step(self, data_batch, optimizer, **kwargs):
        """Supports `optimizer` as dict with keys 'model', 'discriminator_g', 'discriminator_s'."""
        # resolve optimizers
        if isinstance(optimizer, dict):
            opt_g = optimizer.get('model', None)
            opt_dg = optimizer.get('discriminator_g', None)
            opt_ds = optimizer.get('discriminator_s', None)
            # fallbacks
            if opt_g is None:
                opt_g = list(optimizer.values())[0]
            if opt_dg is None:
                opt_dg = list(optimizer.values())[-1]
            if opt_ds is None:
                opt_ds = opt_dg
        else:
            opt_g = optimizer
            opt_dg = optimizer
            opt_ds = optimizer

        # ---------------- G step ----------------
        self.train()
        # enable G params, freeze D params
        for p in self.get_model().parameters():
            p.requires_grad = True
        for p in self.discriminator_g.parameters():
            p.requires_grad = False
        for p in self.discriminator_s.parameters():
            p.requires_grad = False

        losses_g = self.forward_train(**data_batch)
        loss_g, log_vars_g = self._parse_losses(losses_g)

        assert isinstance(loss_g, torch.Tensor) and loss_g.requires_grad

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # ---------------- Dg step ----------------
        for p in self.get_model().parameters():
            p.requires_grad = False
        for p in self.discriminator_g.parameters():
            p.requires_grad = True

        losses_dg = self.forward_train_d(**data_batch)
        loss_dg, log_vars_dg = self._parse_losses({'loss_d_g': losses_dg['loss_d_g']})
        
        opt_dg.zero_grad()
        loss_dg.backward()
        opt_dg.step()

        # ---------------- Ds step ----------------
        for p in self.discriminator_g.parameters():
            p.requires_grad = False
        for p in self.discriminator_s.parameters():
            p.requires_grad = True

        losses_dg = self.forward_train_d(**data_batch)   # <--- recompute
        loss_ds, log_vars_ds = self._parse_losses({'loss_d_s': losses_dg['loss_d_s']})
        # print(loss_ds)
        opt_ds.zero_grad()
        loss_ds.backward()
        opt_ds.step()

        # ---------------- merge logs ----------------
        log_vars = OrderedDict()
        for k, v in log_vars_g.items():
            log_vars[f'G/{k}'] = v
        for k, v in log_vars_dg.items():
            log_vars[f'Dg/{k}'] = v
        for k, v in log_vars_ds.items():
            log_vars[f'Ds/{k}'] = v

        # batch size for runner
        try:
            batch_size = data_batch['img'].data[0].size(0)
        except Exception:
            batch_size = data_batch['img'].size(0) if hasattr(data_batch['img'], 'size') else 1

        return dict(loss=loss_g + loss_dg + loss_ds, log_vars=log_vars, num_samples=batch_size)
