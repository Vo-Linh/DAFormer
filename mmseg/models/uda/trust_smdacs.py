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


@UDA.register_module()
class TrustAwareSMDACS(SMDACS):
    def __init__(self, trust_update_interval=100, **cfg):
        super(TrustAwareSMDACS, self).__init__(**cfg)
        # mode_debug= Fasle
        self.trust_score = None
        self.title_trst_weight = None
        self.trust_update_interval = trust_update_interval  # Update trust score every X iterations
        
        # Accumulation variables for periodic trust score update
        self.accumulated_accuracy = None  # Sum of accuracies for each class
        self.accumulated_mask = None      # Sum of masks (count) for each class
        self.last_trust_update_iter = 0   # Track when trust was last updated
        
        # self.coefficent = coeffient # Coefficient for trust weight adjustment
        
        # Initialize tracking variables
        self.tracking_metrics = {
            'class_accuracies': [],
            'trust_weight_stats': [],
            'pseudo_weight_stats': [],
            'iterations': [],
        }
        
    def compute_trust_weight(self, pseudo_label: torch.Tensor, accuracies: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel accuracy weights (trust) from pseudo labels and class-wise accuracy.

        Args:
            pseudo_label (torch.Tensor): Tensor of shape (B, H, W) containing predicted class indices.
            accuracies (torch.Tensor): Tensor of shape (B, 1, C) containing per-class accuracy scores.

        Returns:
            torch.Tensor: Tensor of shape (B, H, W) containing per-pixel trust values.
        """
        B, H, W = pseudo_label.shape
        num_classes = accuracies.size(-1)
        if torch.any(pseudo_label >= self.num_classes) or torch.any(pseudo_label < 0):
            raise ValueError(
                f"Invalid pseudo_label values found: min={pseudo_label.min()}, max={pseudo_label.max()}, num_classes={self.num_classes}")
        # Ensure accuracy tensor shape is (B, C, 1, 1) for broadcasting
        if accuracies.size(0) == 1:
            # Broadcast to (B, C, 1, 1)
            accuracy_expanded = accuracies.squeeze(
                1).repeat(B, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            accuracy_expanded = accuracies.squeeze(
                1).unsqueeze(-1).unsqueeze(-1)

        # One-hot encode pseudo labels and permute to (B, C, H, W)
        pseudo_onehot = F.one_hot(
            pseudo_label, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Multiply one-hot labels with class accuracy and sum over class dimension
        return (pseudo_onehot * accuracy_expanded).sum(dim=1)  # (B, H, W)

    def compute_accuracy(self, pseudo_label, target_gt_semantic_seg, num_classes):
        """
        Compute accuracy between pseudo-labels and ground truth labels.

        Args:
            pseudo_label (Tensor): Pseudo-labels (B x H x W) or (H x W).
            target_gt_semantic_seg (Tensor): Ground truth labels (B x 1 x H x W) or (1 x H x W).
            num_classes (int): Total number of classes.

        Returns:
            Tensor: Accuracy for each class, with shape (B, 1, C) or (1, 1, C).
        """
        # Handle input shape without batch dimension
        if pseudo_label.dim() == 2:
            pseudo_label = pseudo_label.unsqueeze(0)  # (1, H, W)
        if target_gt_semantic_seg.dim() == 3:
            target_gt_semantic_seg = target_gt_semantic_seg.unsqueeze(
                0)  # (1, 1, H, W)

        batch_size = pseudo_label.shape[0]
        device = pseudo_label.device
        _, _, H, W = target_gt_semantic_seg.shape

        all_class_accuracies = torch.zeros(
            batch_size, num_classes, device=device)

        for i in range(batch_size):
            pseudo_flat = pseudo_label[i].view(-1)
            gt_flat = target_gt_semantic_seg[i].squeeze(0).view(-1)

            for class_id in range(num_classes):
                pseudo_class_mask = (pseudo_flat == class_id)
                gt_class_mask = (gt_flat == class_id)

                correct_predictions = torch.sum(
                    pseudo_class_mask & gt_class_mask)
                total_gt_pixels = torch.sum(gt_class_mask)

                if total_gt_pixels > 0:
                    all_class_accuracies[i, class_id] = correct_predictions.float(
                    ) / total_gt_pixels

        return all_class_accuracies.unsqueeze(1)  # Shape: (B, 1, C)

    def _accumulate_accuracy_and_mask(self, accuracy_tensor):
        """
        Accumulate accuracy and mask for periodic trust score update.
        
        Args:
            accuracy_tensor (torch.Tensor): Tensor of shape (B, 1, C) containing per-class accuracy scores.
        """
        device = accuracy_tensor.device
        num_classes = accuracy_tensor.size(-1)
        
        # Initialize accumulation tensors if not exist
        if self.accumulated_accuracy is None:
            self.accumulated_accuracy = torch.zeros((1, 1, num_classes), device=device)
            self.accumulated_mask = torch.zeros((1, 1, num_classes), device=device)
        
        # Create mask for classes that have non-zero accuracy (indicating they were present in GT)
        mask = (accuracy_tensor > 0).float()
        
        # Accumulate accuracy and mask
        self.accumulated_accuracy += accuracy_tensor.sum(dim=0)  # Sum across batch
        self.accumulated_mask += mask.sum(dim=0)  # Count across batch
        
    def _update_trust_score(self, alpha=0.8):
        """
        Update trust score using accumulated accuracy and mask data.
        
        Args:
            alpha (float): EMA decay factor for trust score update.
        """
        if self.accumulated_accuracy is None or self.accumulated_mask is None:
            return
        
        # Compute average accuracy for each class
        # Avoid division by zero
        mask_nonzero = self.accumulated_mask > 0
        avg_accuracy = torch.zeros_like(self.accumulated_accuracy)
        avg_accuracy[mask_nonzero] = (self.accumulated_accuracy[mask_nonzero] / 
                                     self.accumulated_mask[mask_nonzero])
        
        # Initialize or update trust score
        if self.trust_score is None:
            self.trust_score = avg_accuracy.clone()
            self.title_trst_weight = "Initial trust weight from accumulated data"
        else:
            # Only update classes that have data (mask > 0)
            update_mask = (self.accumulated_mask > 0).float()
            self.trust_score = ((1 - alpha) * update_mask * avg_accuracy + 
                               alpha * self.trust_score)
            self.title_trst_weight = "Updated trust weight from accumulated data"
        
        # Reset accumulation variables
        self.accumulated_accuracy.zero_()
        self.accumulated_mask.zero_()
        self.last_trust_update_iter = self.local_iter
        
        # Log the update
        trust_score_list = self.trust_score.squeeze().cpu().tolist()
        mmcv.print_log(f'Trust score updated at iteration {self.local_iter}: {trust_score_list}', 'mmseg')

    def _should_update_trust_score(self):
        """
        Check if trust score should be updated based on iteration interval.
        
        Returns:
            bool: True if trust score should be updated.
        """
        return (self.local_iter - self.last_trust_update_iter) >= self.trust_update_interval

    def _track_metrics(self, avg_accuracy_tensor, trust_weight, pseudo_weight):
        """Track metrics during training"""
        # Track class accuracies
        if avg_accuracy_tensor is not None:
            class_acc = avg_accuracy_tensor.squeeze().cpu().numpy()
            self.tracking_metrics['class_accuracies'].append({
                'iteration': self.local_iter,
                'accuracies': class_acc.tolist(),
                'mean_accuracy': float(class_acc.mean()),
                'std_accuracy': float(class_acc.std())
            })
        
        # Track trust weight statistics
        if trust_weight is not None:
            trust_stats = {
                'iteration': self.local_iter,
                'mean': float(trust_weight.mean().cpu()),
                'std': float(trust_weight.std().cpu()),
                'min': float(trust_weight.min().cpu()),
                'max': float(trust_weight.max().cpu()),
                'median': float(trust_weight.median().cpu())
            }
            self.tracking_metrics['trust_weight_stats'].append(trust_stats)
        
        # Track pseudo weight statistics
        if pseudo_weight is not None:
            pseudo_stats = {
                'iteration': self.local_iter,
                'mean': float(pseudo_weight.mean().cpu()),
                'std': float(pseudo_weight.std().cpu()),
                'min': float(pseudo_weight.min().cpu()),
                'max': float(pseudo_weight.max().cpu()),
                'median': float(pseudo_weight.median().cpu())
            }
            self.tracking_metrics['pseudo_weight_stats'].append(pseudo_stats)
        
        self.tracking_metrics['iterations'].append(self.local_iter)

    def _log_tracking_metrics(self, log_vars):
        """Add tracking metrics to log variables"""
        if len(self.tracking_metrics['class_accuracies']) > 0:
            latest_acc = self.tracking_metrics['class_accuracies'][-1]
            log_vars['mean_class_accuracy'] = latest_acc['mean_accuracy']
            log_vars['std_class_accuracy'] = latest_acc['std_accuracy']
        
        if len(self.tracking_metrics['trust_weight_stats']) > 0:
            latest_trust = self.tracking_metrics['trust_weight_stats'][-1]
            log_vars['trust_weight_mean'] = latest_trust['mean']
            log_vars['trust_weight_std'] = latest_trust['std']
        
        if len(self.tracking_metrics['pseudo_weight_stats']) > 0:
            latest_pseudo = self.tracking_metrics['pseudo_weight_stats'][-1]
            log_vars['pseudo_weight_mean'] = latest_pseudo['mean']
            log_vars['pseudo_weight_std'] = latest_pseudo['std']

    def _save_tracking_data(self):
        """Save tracking data to file"""
        if hasattr(self, 'train_cfg') and 'work_dir' in self.train_cfg:
            tracking_dir = os.path.join(self.train_cfg['work_dir'], 'tracking')
            os.makedirs(tracking_dir, exist_ok=True)
            
            # Save as numpy files for easy loading
            np.save(os.path.join(tracking_dir, 'tracking_metrics.npy'), 
                   self.tracking_metrics, allow_pickle=True)
            
            # Also save as text for human readability
            with open(os.path.join(tracking_dir, 'tracking_log.txt'), 'a') as f:
                f.write(f"Iteration {self.local_iter}:\n")
                if len(self.tracking_metrics['class_accuracies']) > 0:
                    latest_acc = self.tracking_metrics['class_accuracies'][-1]
                    f.write(f"  Accumulate accuracy: {self.accumulated_accuracy} \n")
                    f.write(f"  Accumulate mask: {self.accumulated_mask}\n")
                    f.write(f"  Class Accuracies: {latest_acc['accuracies']}\n")
                    f.write(f"  Mean Accuracy: {latest_acc['mean_accuracy']:.4f}\n")
                    trust_score_list = self.trust_score.squeeze().cpu().tolist()
                    f.write(f"  S_tr: {trust_score_list}\n")
                if len(self.tracking_metrics['trust_weight_stats']) > 0:
                    latest_trust = self.tracking_metrics['trust_weight_stats'][-1]
                    f.write(f"  Trust Weight - Mean: {latest_trust['mean']:.4f}, Std: {latest_trust['std']:.4f}\n")
                if len(self.tracking_metrics['pseudo_weight_stats']) > 0:
                    latest_pseudo = self.tracking_metrics['pseudo_weight_stats'][-1]
                    f.write(f"  Pseudo Weight - Mean: {latest_pseudo['mean']:.4f}, Std: {latest_pseudo['std']:.4f}\n")
                
                f.write("\n")

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
            self.title_trst_weight = "Initial trust weight"
            self.trust_score = torch.ones(
                (1, 1, self.num_classes), device=device)
        elif self.local_iter > 0:
            self._update_ema(self.local_iter)

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
        accuracy_tensor_list = []
        avg_accuracy_tensor = None
        
        for i in range(batch_size):
            has_labels = target_img_metas[i].get('with_labels', False)
            use_gt.append(has_labels)

            # Override pseudo-labels with ground truth when available
            if has_labels:
                accuracy_tensor = self.compute_accuracy(
                    pseudo_label[i],
                    target_gt_semantic_seg[i],
                    num_classes=self.num_classes
                )
                accuracy_tensor_list.append(accuracy_tensor)

                pseudo_weight[i] = torch.ones_like(
                    pseudo_weight[i], device=device)
                pseudo_label[i] = target_gt_semantic_seg[i].squeeze(0)
                self.debug_gt += 1

        gt_pixel_weight = torch.ones_like(pseudo_weight, device=device)

        # Handle accuracy accumulation and trust score update
        if accuracy_tensor_list:
            # Stack and compute average accuracy for current batch
            avg_accuracy_tensor = torch.stack(accuracy_tensor_list).mean(dim=0)
            
            # Accumulate accuracy and mask for periodic update
            self._accumulate_accuracy_and_mask(torch.stack(accuracy_tensor_list))
            
            # Check if it's time to update trust score
            if self._should_update_trust_score():
                self._update_trust_score(alpha=0.99)
        else:
            # Create zero accuracy tensor when no ground truth is available
            avg_accuracy_tensor = torch.zeros((1, 1, self.num_classes), device=device)

        # Compute trust weight using current trust score
        trust_weight = self.compute_trust_weight(
            pseudo_label_keep, self.trust_score)
        
        # pseudo_weight = pseudo_weight * (trust_weight ** self.coefficent) 
        pseudo_label = pseudo_label * trust_weight


        # Track metrics here - after all weights are computed
        self._track_metrics(avg_accuracy_tensor, trust_weight, pseudo_weight)

        # Apply mixing strategy
        mixed_img, mixed_lbl = [], []
        mix_masks = get_class_masks(gt_semantic_seg)
        if self.local_iter % self.debug_img_interval == 0:
            before_update_pseudo_weight = pseudo_weight.clone()
            
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]

            # Mix source and target images/labels
            img_i, lbl_i = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i]))
            )
            mixed_img.append(img_i)
            mixed_lbl.append(lbl_i)

            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))

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

        # Add tracking metrics to log variables
        self._log_tracking_metrics(log_vars)

        # Save tracking data periodically
        if accuracy_tensor_list:
            self._save_tracking_data()

        # Visualization for debugging
        if self.local_iter % self.debug_img_interval == 0:
            self._save_debug_images(
                img, target_img, mixed_img,
                gt_semantic_seg, target_gt_semantic_seg, mixed_lbl,
                pseudo_label, pseudo_label_keep, pseudo_weight, mix_masks, trust_weight,
                means, stds, batch_size, use_gt, before_update_pseudo_weight
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