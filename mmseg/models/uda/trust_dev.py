import math
import os
import random
import time
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg

from .smdacs import SMDACS

from mmseg.datasets import build_dataset, build_dataloader
from mmseg.core import eval_metrics

from mmcv.utils import print_log



@UDA.register_module()
class TrustAwareDev(SMDACS):
    def __init__(self, trust_update_interval=100, coefficient=1, **cfg):
        super(TrustAwareDev, self).__init__(**cfg)
        # mode_debug= Fasle
        self.trust_score = None
        self.trust_update_interval = trust_update_interval  # Update trust score every X iterations
        
        self.coefficient = coefficient # Coefficient for trust weight adjustment
        cfg['target_dataset']['type']=cfg['target_dataset']['type'].replace('SM', '')
        # print_log(cfg['target_dataset']['pipeline'], 'mmseg')

        self.target_dataset = build_dataset(cfg['target_dataset'], dict(test_mode=False))
        self.target_dataloader = build_dataloader(
            self.target_dataset,
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=False,
            shuffle=False)
    def target_evaluate(self, dataloader, model, **kwargs):
        model.eval()
        results = []
        gt_segs = []

        dataset = dataloader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))

        for data in dataloader:
            # Extract image metadata and image tensors
            img_metas = self._extract_data(data['img_metas'])
            imgs = self._extract_data(data['img'])

            # Extract and preprocess ground-truth segmentations
            gt_batch = self._extract_data(data['gt_semantic_seg'])
            for gt in gt_batch:
                gt = gt.squeeze().unsqueeze(0).cpu().numpy()  # Shape: (1, H, W)
                gt_segs.append(gt)

            # Move data to device
            device = next(model.parameters()).device
            imgs = imgs.to(device)
            data['img'] = [imgs]
            data['img_metas'] = img_metas

            with torch.no_grad():
                logits = model.encode_decode(imgs, img_metas)
                probs = torch.softmax(logits.detach(), dim=1)
                _, preds = torch.max(probs, dim=1)

                del logits, probs
                torch.cuda.empty_cache()

            results.append(preds.cpu().numpy())

            prog_bar.update(len(preds))

        # Evaluate and return metrics
        eval_results = eval_metrics(
            results,
            gt_segs,
            num_classes=len(dataset.CLASSES),
            ignore_index=dataset.ignore_index,
            metric='mIoU'
        )
        return eval_results

    def _extract_data(self, data_field):
        """
        Helper method to extract data from mmcv.parallel.DataContainer
        """
        if isinstance(data_field, mmcv.parallel.DataContainer):
            return data_field.data[0]
        return [d.data[0] if isinstance(d, mmcv.parallel.DataContainer) else d for d in data_field]        
    # def target_evaluate(self, target_dataloader, model, **kwargs):
    #     model.eval()
    #     results = []
    #     gt_segs = []
        
    #     dataset = target_dataloader.dataset
    #     prog_bar = mmcv.ProgressBar(len(dataset))

    #     for i, data in enumerate(target_dataloader):
    #         img_metas = data['img_metas']
    #         imgs = data['img']

    #         if isinstance(img_metas, mmcv.parallel.DataContainer):
    #             img_metas = img_metas.data[0]
    #         else:
    #             img_metas = [
    #                 im.data[0] if isinstance(im, mmcv.parallel.DataContainer) else im
    #                 for im in img_metas
    #             ]
    #         if isinstance(imgs, mmcv.parallel.DataContainer):
    #             imgs = imgs.data[0]
    #         else:
    #             imgs = [im.data[0] if isinstance(im, mmcv.parallel.DataContainer) else im for im in imgs]
            
    #         img_seg = data['gt_semantic_seg']
    #         if isinstance(img_seg, mmcv.parallel.DataContainer):
    #             img_seg = img_seg.data[0]  # List[Tensor]

    #         # Process each image in batch
    #         for seg in img_seg:
    #             seg = seg.squeeze().unsqueeze(0).cpu().numpy()  # Shape: (1, 512, 512)
    #             gt_segs.append(seg)

    #         data['img_metas'] = img_metas
    #         device = next(model.parameters()).device
    #         imgs = imgs.to(device)
    #         data['img'] = [imgs]
        
    #         # mmcv.print_log(imgs[0].shape, 'mmseg')
    #         # mmcv.print_log(len(img_metas), 'mmseg')
            
    #         # mmcv.print_log(data['img_metas'][0], 'mmseg')


    #         # ema_logits = self.get_ema_model().encode_decode(imgs, img_metas)
    #         # print("Model device:", next(model.parameters()).device)
    #         # print("Image tensor device:", img_seg.device)
    #         with torch.no_grad():
    #             # result = model(return_loss=False, **data)
    #             #  = self.get_ema_model().encode_decode()
    #             ema_logits = model.encode_decode(imgs, img_metas)
    #             ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
    #             pseudo_prob, result = torch.max(ema_softmax, dim=1)
    #             del ema_logits, ema_softmax
    #             torch.cuda.empty_cache()
    #         batch_size = len(result)
    #         for _ in range(batch_size):
    #             prog_bar.update()
    #         results.append(result.cpu().numpy())

    #     # outputs = single_gpu_test(model, target_dataloader, **kwargs)
    #     # mmcv.print_log(f"results shape: {np.array(results).shape}", 'mmseg')
    #     # mmcv.print_log(f"gt_segs shape: {np.array(gt_segs).shape}", 'mmseg')
    #     eval_results= eval_metrics(results, gt_segs, len(dataset.CLASSES), dataset.ignore_index, 'mIoU')
    #     return eval_results
    def update_trust_score(self, acc, alpha=0.99):
        """Update the trust score based on the accuracy."""
        if isinstance(acc, np.ndarray):
            acc = torch.from_numpy(acc).float()

            acc = acc.to(self.trust_score.device)  # make sure it's on the same device

        self.trust_score = ((1 - alpha)* acc + alpha * self.trust_score)
        mmcv.print_log(f'Updated trust score at {self.local_iter}: {self.trust_score}', 'mmseg')
    def compute_trust_weight(self, pseudo_label: torch.Tensor, coefficient=1) -> torch.Tensor:
        """
        Compute per-pixel trust weights from pseudo labels using class-wise trust scores.

        Args:
            pseudo_label (torch.Tensor): shape (B, H, W), with class indices

        Returns:
            torch.Tensor: shape (B, H, W), containing per-pixel trust values
        """
        B, H, W = pseudo_label.shape

        if torch.any(pseudo_label >= self.num_classes) or torch.any(pseudo_label < 0):
            raise ValueError(
                f"Invalid pseudo_label values found: min={pseudo_label.min()}, max={pseudo_label.max()}, num_classes={self.num_classes}"
            )

        # Make sure trust_score is a 1D tensor: [C]
        if isinstance(self.trust_score, (list, tuple)):
            trust_score = torch.tensor(self.trust_score, dtype=torch.float32, device=pseudo_label.device).clone()
        else:
            trust_score = self.trust_score.to(pseudo_label.device).clone()

        if trust_score.ndim != 1:
            trust_score = trust_score.view(-1)  # flatten to [C]
        trust_score = trust_score ** coefficient  # Apply coefficient to trust score
        # Use indexing to assign per-pixel trust weights
        trust_weight = trust_score[pseudo_label]  # shape: [B, H, W]

        # print(f'Trust weight shape: {trust_weight.shape}')
        return trust_weight



    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas, target_gt_semantic_seg, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict.
            gt_semantic_seg (Tensor): Semantic segmentation masks for source.
            target_img (Tensor): Target domain images.
            target_img_metas (list[dict]): List of target image info dict.
            target_gt_semantic_seg (Tensor): Target domain segmentation masks.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.title_trst_weight = "Used old trust weight"
        log_vars = {}
        batch_size, C, W, H = img.shape
        device = img.device

        # Initialize/update EMA model
        if self.local_iter == 0:
            self._init_ema_weights()
            self.trust_score = torch.ones(
                (1, 1, self.num_classes), device=device)
        elif self.local_iter > 0:
            self._update_ema(self.local_iter)
        
        if  self.local_iter > 1 and self.local_iter % self.trust_update_interval == 0:
            print(f'Updating trust score at iteration {self.local_iter}')
            self._set_dropout_eval()
            eval_result=self.target_evaluate(self.target_dataloader, self.get_ema_model(), efficient_test=False)
            Acc = eval_result['Acc']
            log_vars.update(add_prefix(eval_result, 'tar_eval'))
            self.update_trust_score(Acc)

        # Get mean & std for normalization operations
        means, stds = get_mean_std(img_metas, device)

        # Set up strong augmentation parameters
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)

        # Print gradient magnitude if requested
        if self.print_grad_magnitude:
            self._log_grad_magnitude('Seg')

        # ImageNet feature distance calculation
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(
                img, gt_semantic_seg, src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))

            if self.print_grad_magnitude:
                self._log_grad_magnitude('Fdist', base_grads=[p.grad.detach().clone() for p in
                                                              self.get_model().backbone.parameters() if p.grad is not None])

        # Set dropout layers to eval mode for pseudo-label generation
        self._set_dropout_eval()

        # Generate pseudo-labels
        ema_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        del ema_logits, ema_softmax
        torch.cuda.empty_cache()
        
        # Calculate pseudo-label confidence
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * \
            torch.ones(pseudo_prob.shape, device=device)

        # Store original pseudo-labels
        pseudo_label_keep = pseudo_label.clone()

        # Apply ignore regions if specified
        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0

        # Compute accuracy if ground truth is available and accumulate
        use_gt = []
        
        for i in range(batch_size):
            has_labels = target_img_metas[i].get('with_labels', False)
            use_gt.append(has_labels)

            # Override pseudo-labels with ground truth when available
            if has_labels:
                pseudo_weight[i] = torch.ones_like(
                    pseudo_weight[i], device=device)
                pseudo_label[i] = target_gt_semantic_seg[i].squeeze(0)

        gt_pixel_weight = torch.ones_like(pseudo_weight, device=device)


        # Compute trust weight using current trust score
        trust_weight = self.compute_trust_weight(
            pseudo_label_keep, self.coefficient)
        
        pseudo_weight = pseudo_weight * trust_weight 
        log_vars.update({
            'pseudo_weight_mean': pseudo_weight.mean().item(),
        })
        if self.local_iter % self.debug_img_interval == 0:
            before_update_pseudo_weight = pseudo_weight.clone()
            
        # # pseudo_weight = pseudo_weight * trust_weight


        # Track metrics here - after all weights are computed
        # self._track_metrics(avg_accuracy_tensor, trust_weight, pseudo_weight)

        # Apply mixing strategy
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)


        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            # pseudo_weight[i] = torch.ones((pseudo_weight[i].shape), device=dev)
        
        # Concatenate mixed data
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        # # Save tracking data periodically
        # if accuracy_tensor_list:
        #     self._save_tracking_data()

        # Visualization for debugging
        if self.local_iter % self.debug_img_interval == 0:
            self._save_debug_images(
                img, target_img, mixed_img,
                gt_semantic_seg, target_gt_semantic_seg, mixed_lbl,
                pseudo_label, pseudo_label_keep, pseudo_weight, mix_masks, trust_weight,
                means, stds, batch_size, use_gt, before_update_pseudo_weight,
            )

        self.local_iter += 1
        return log_vars

    def _set_dropout_eval(self):
        """Set dropout and DropPath layers to eval mode"""
        for m in self.get_ema_model().modules():
            if isinstance(m, (_DropoutNd, DropPath)):
                m.training = False

    def _log_grad_magnitude(self, name, base_grads=None):
        """Log the magnitude of gradients"""
        params = self.get_model().backbone.parameters()
        grads = [p.grad.detach().clone() for p in params if p.grad is not None]

        if base_grads is not None:
            grads = [g2 - g1 for g1, g2 in zip(base_grads, grads)]

        grad_mag = sum(g.norm().item() for g in grads) / len(grads)
        mmcv.print_log(f'{name} Grad.: {grad_mag:.4f}', 'mmseg')

    def _save_debug_images(self, img, target_img, mixed_img, gt_semantic_seg,
                           target_gt_semantic_seg, mixed_lbl, pseudo_label,
                           pseudo_label_keep, pseudo_weight, mix_masks, trust_weight,
                           means, stds, batch_size, use_gt, before_update_pseudo_weight):
        """Save debug visualization images"""
        out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
        os.makedirs(out_dir, exist_ok=True)

        # Denormalize images for visualization
        vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
        vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
        vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)

        if trust_weight is not None:
            trust_weight_vis = trust_weight.detach().cpu().clamp(0, 1)
        else:
            trust_weight_vis = [None] * batch_size
        for j in range(batch_size):
            rows, cols = 2, 6
            _, axs = plt.subplots(
                rows, cols,
                figsize=(3 * cols, 3 * rows),
                gridspec_kw={
                    'hspace': 0.1, 'wspace': 0, 'top': 0.95,
                    'bottom': 0, 'right': 1, 'left': 0
                },
            )

            # Plot source domain data
            subplotimg(axs[0][0], vis_img[j], 'Source Image')
            subplotimg(axs[0][1], gt_semantic_seg[j],
                       'Source Seg GT', cmap=self.cmap)

            # Plot target domain data
            subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
            subplotimg(axs[1][1], target_gt_semantic_seg[j],
                       'Target Seg GT', cmap=self.cmap)

            # Plot pseudo-labels
            if use_gt[j]:
                subplotimg(axs[0][2], pseudo_label[j],
                           'Target Seg Pseudo GT', cmap=self.cmap)
                subplotimg(axs[1][2], pseudo_label_keep[j],
                           'Target Seg Pseudo Gen', cmap=self.cmap)
            else:
                subplotimg(axs[0][2], pseudo_label[j],
                           'Target Seg Pseudo', cmap=self.cmap)

            # Plot mixed data
            subplotimg(axs[0][3], vis_mixed_img[j], 'Mixed Image')
            subplotimg(axs[1][3], mix_masks[j][0], 'Domain Mask', cmap='gray')
            subplotimg(axs[0][4], mixed_lbl[j],
                       'Seg Mixed Targ', cmap=self.cmap)
            subplotimg(axs[1][4], pseudo_weight[j],
                       'Final Pseudo W.', vmin=0, vmax=1)
            
            subplotimg(axs[0][5], before_update_pseudo_weight[j],
                           "DAFormer Pseudo W.", vmin=0, vmax=1)
            if trust_weight_vis is not None:
                subplotimg(axs[1][5], trust_weight_vis[j],
                           self.title_trst_weight, vmin=0, vmax=1)

            # Plot additional debug info if available
            if hasattr(self, 'debug_fdist_mask') and self.debug_fdist_mask is not None:
                subplotimg(axs[0][6], self.debug_fdist_mask[j]
                           [0], 'FDist Mask', cmap='gray')
            if hasattr(self, 'debug_gt_rescale') and self.debug_gt_rescale is not None:
                subplotimg(axs[1][6], self.debug_gt_rescale[j],
                           'Scaled GT', cmap=self.cmap)

            # Turn off axes for all subplots
            for ax in axs.flat:
                ax.axis('off')

            # Save the figure
            plt.savefig(os.path.join(
                out_dir, f'{(self.local_iter + 1)}_{j}.png'))
            plt.close()