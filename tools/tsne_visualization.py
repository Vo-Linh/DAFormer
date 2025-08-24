

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Modification of config and checkpoint to support legacy models

import argparse
import os
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import time
def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.val.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(
            type='MultiScaleFlipAug',
            img_scale=tuple([512, 512]),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
                dict(type='Collect', keys=['img', 'gt_semantic_seg']),
            ]
        )
    ]
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    if 'vib_params' not in cfg.model.decode_head:
        cfg.model.decode_head.vib_params = None  
    return cfg

import torch
import re

def convert_checkpoint(old_ckpt_path):
    ckpt = torch.load(old_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k

        # ---- Patch Embeddings ----
        if "patch_embed1" in k:
            new_k = k.replace("patch_embed1", "patch_embed.0")
        elif "patch_embed2" in k:
            new_k = k.replace("patch_embed2", "patch_embed.1")
        elif "patch_embed3" in k:
            new_k = k.replace("patch_embed3", "patch_embed.2")
        elif "patch_embed4" in k:
            new_k = k.replace("patch_embed4", "patch_embed.3")

        # ---- Block renaming ----
        # old: block1.0.norm1 -> new: layers.0.blocks.0.norm1
        match = re.match(r"model\.backbone\.block(\d+)\.(\d+)\.(.*)", k)
        if match:
            stage, block, suffix = match.groups()
            stage_idx = int(stage) - 1   # block1 -> layers.0
            new_k = f"model.backbone.layers.{stage_idx}.blocks.{block}.{suffix}"

        # ---- Norm at end of stage ----
        match_norm = re.match(r"model\.backbone\.norm(\d+)\.(.*)", k)
        if match_norm:
            stage, suffix = match_norm.groups()
            stage_idx = int(stage) - 1
            new_k = f"model.backbone.layers.{stage_idx}.norm.{suffix}"

        new_state_dict[new_k] = v

    # Save new checkpoint
    ckpt["state_dict"] = new_state_dict
    return ckpt


def single_gpu_infer_features(model, dataloader, args=None):
    feats_all = []
    gts_all = []
    if args.num_images <= len(dataloader):
        print(f"Visualizing features for {args.num_images} images.")
    else:
        print(f"Warning: Number of images {args.num_images} exceeds dataset size {len(dataloader)}. Visualizing all images instead.")
        args.num_images = len(dataloader)
    prog_bar = mmcv.ProgressBar(args.num_images)

    for i, data in enumerate(dataloader):
        if i >= args.num_images:
            break
        with torch.no_grad():
            result = model.module.encode_decode(data['img'][0].cuda(), data['img_metas'][0])
            if isinstance(result, (tuple, list)):
                result = result[0]

            # features: [C, H, W], gt: [H_gt, W_gt]
            if result.dim() == 4:
                result = result.squeeze(0)
            C, H, W = result.shape
            result = result.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
            
            feats_all.append(result)  # You may need to adjust this if your model returns features differently
            gts_all.append(data['gt_semantic_seg'][0].view(-1).cpu().numpy())
            torch.cuda.empty_cache() 
        prog_bar.update()
    return feats_all, gts_all

def extract_and_tsne_visualize(all_feats, all_labels, palette, class_names, out_path=None, seed=42, max_samples=300):
    # Merge all batches
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Total feature points: {all_feats.shape[0]}, Total labels: {all_labels.shape[0]}")

    sampled_feats = []
    sampled_labels = []
    np.random.seed(seed)

    # Sample features by class
    for cls_id in np.unique(all_labels):
        if cls_id >= len(class_names):
            print(f"Warning: Class ID {cls_id} exceeds class names length {len(class_names)}")
            continue
        elif cls_id == 0:  # skip background
            print(f"Skipping class ID {cls_id} (background)")
            continue

        cls_mask = (all_labels == cls_id)
        indices = np.where(cls_mask)[0]

        if len(indices) > max_samples:
            sampled_idx = np.random.choice(indices, size=max_samples, replace=False)
        else:
            sampled_idx = indices

        sampled_feats.append(all_feats[sampled_idx])
        sampled_labels.append(np.full(sampled_idx.shape, cls_id))

    # Combine sampled features
    feats = np.concatenate(sampled_feats, axis=0)
    labels = np.concatenate(sampled_labels, axis=0)

    # Run t-SNE
    print(f"Running t-SNE on {feats.shape[0]} feature points...")
    time_start = time.time()
    tsne = TSNE(n_jobs=30, n_iter=500, perplexity=30, random_state=seed)  # lower perplexity
    embedded = tsne.fit_transform(feats)
    print(f"t-SNE completed in {time.time() - time_start:.2f} seconds")

    indices = np.arange(len(embedded))
    np.random.shuffle(indices)
    
    embedded = embedded[indices]
    labels = labels[indices]
    # Visualization
    palette[0] = [0, 0, 0]  # background black
    palette = np.array(palette)
    plt.figure(figsize=(5,5), dpi=500)

    for cls_id in range(len(class_names)):
        mask = labels == cls_id
        if np.any(mask):
            plt.scatter(
                embedded[mask, 0], embedded[mask, 1],
                s=5, alpha=0.7, color=palette[cls_id] / 255.0
            )

    plt.axis('off')

    # Save or show
    if out_path:
        out = f"{out_path.split('.png')[0]}_{seed}.png"
        plt.savefig(out, dpi=1000, bbox_inches='tight')
        print(f"Saved t-SNE figure to {out}")
    else:
        plt.show()
    plt.close() 



def tsne_visualize_2domains(
    all_feats, all_labels, class_id, domain_names, out_path=None, max_samples=300, seed=42
):

    sampled_feats = []
    sampled_labels = []
    idx=0
    for feats, labels in zip(all_feats, all_labels):
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)
        print(f"Total feature points: {feats.shape[0]}, Total labels: {labels.shape[0]}")
        for cls_id in class_id:
            cls_mask = (labels == cls_id)
            indices = np.where(cls_mask)[0]
            if len(indices) > max_samples:
                sampled_idx = np.random.choice(indices, size=max_samples, replace=False)
            else:
                sampled_idx = indices
            sampled_feats.append(feats[sampled_idx])
            sampled_labels.append(np.full_like(sampled_idx, idx))
        idx += 1
    

    feats = np.concatenate(sampled_feats, axis=0)
    labels = np.concatenate(sampled_labels, axis=0)


    print(f"Running t-SNE on {feats.shape[0]} feature points with seed {seed}...")
    time_start = time.time()
    tsne = TSNE(n_jobs=30, n_iter=500, perplexity=30, random_state=seed)
    embedded = tsne.fit_transform(feats)
    print(f"t-SNE completed in {time.time() - time_start:.2f} seconds")

    # Shuffle to avoid plotting sequence artifacts
    indices = np.arange(len(feats))
    np.random.shuffle(indices)
    embedded = embedded[indices]
    labels = labels[indices]

    # Visualization
    palette = [[0, 150, 0], [255, 200, 150]]
    print(f"Palette: {palette}")
    palette = np.array(palette)
    print(f"Labels corresponding to classes: {np.unique(labels)}")
    print(f"Class names: {domain_names}")
    
    
    plt.figure(figsize=(3, 3), dpi=500)
    for cls_id in range(len(domain_names)):
        mask = labels == cls_id
        if np.any(mask):
            plt.scatter(
                embedded[mask, 0], embedded[mask, 1],
                s=5, alpha=0.7, color=palette[cls_id] / 255.0,
                label=domain_names[cls_id]
            )

    plt.axis('off')

    if out_path:
        out = f"{out_path.split('.png')[0]}_{seed}.png"
        plt.savefig(out, dpi=1000, bbox_inches='tight')
        print(f"Saved t-SNE figure to {out}")
    else:
        plt.show()
    plt.close() 
def parse_args():
    parser = argparse.ArgumentParser(description='MMSegmentation t-SNE Feature Visualization')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show-dir', default='./work_dirs/tsne_vis', help='Directory to save results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num-images', type=int, default=10, help='Number of images to visualize')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.show_dir)

    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    # cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    # print(cfg.model)
    # cfg.model.type = 'EncoderDecoderForSupervised'
    # print(f"Val dataset config: {cfg.data.val}")
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # convert_checkpoint(args.checkpoint)
    wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    
    
    CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
               'agricultural')

    PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
               [159, 129, 183], [0, 255, 0], [255, 195, 128]]

    
    feats_all = []
    gts_all = []
    if os.path.exists(os.path.join(args.show_dir, f'feats_urban.npy')):
        feats_rural = np.load(os.path.join(args.show_dir, f'feats_rural.npy'))
        gts_rural = np.load(os.path.join(args.show_dir, f'gts_rural.npy'))
        feats_urban = np.load(os.path.join(args.show_dir, f'feats_urban.npy'))
        gts_urban = np.load(os.path.join(args.show_dir, f'gts_urban.npy'))
        print(feats_rural.shape, feats_urban.shape)
        feats_all.append(feats_urban)
        gts_all.append(gts_urban)
        feats_all.append(feats_rural)
        gts_all.append(gts_rural)
    else:
        for domain in ['Rural', 'Urban']:
            cfg = update_legacy_cfg(cfg)

            cfg.data.val.data_root = f'/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA/Val/{domain}'
            cfg.data.val.test_mode = False
            cfg.data.val.img_dir = 'images_png'
            cfg.data.val.ann_dir = 'masks_png'
            dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            # print(f"Dataset: {dataset}")
            dataloader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=4,
                shuffle=False)
            if domain == 'Rural':
                feats_rural, gts_rural = single_gpu_infer_features(model, dataloader, args)
                np.save(os.path.join(args.show_dir, f'feats_rural.npy'), feats_rural)
                np.save(os.path.join(args.show_dir, f'gts_rural.npy'), gts_rural)
                feats_all.append(feats_rural)
                gts_all.append(gts_rural)

            elif domain == 'Urban':
                feats_urban, gts_urban = single_gpu_infer_features(model, dataloader, args)
                np.save(os.path.join(args.show_dir, f'feats_urban.npy'), feats_urban)
                np.save(os.path.join(args.show_dir, f'gts_urban.npy'), gts_urban)
                feats_all.append(feats_urban)
                gts_all.append(gts_urban)
        
    out_path = os.path.join(args.show_dir, f'tsne_plot_Urban.png')
    for i in range(3):
        extract_and_tsne_visualize(feats_urban, gts_urban, PALETTE, CLASSES, out_path, seed=i, max_samples=500)
    out_path = os.path.join(args.show_dir, f'tsne_plot_Rural.png')
    for i in range(3):
        extract_and_tsne_visualize(feats_rural, gts_rural, PALETTE, CLASSES, out_path, seed=i, max_samples=500)

    
    for i in range(7):   
        for j in range(3): 
            out_path = os.path.join(args.show_dir, f'{CLASSES[i]}_tsne_plot_2domains.png')
            tsne_visualize_2domains(feats_all, gts_all, [i], ['Rural', 'Urban'], out_path, max_samples=500, seed=j)
    threshold = 0.05
    def count_ge_threshold(array, threshold):
        return np.count_nonzero(np.abs(array) >= threshold)

    feats_rural = np.concatenate(feats_rural, axis=0)
    feats_urban = np.concatenate(feats_urban, axis=0)
    feats_all = np.concatenate(feats_all, axis=0)
    rural_sparsity = 1 - count_ge_threshold(feats_rural, threshold) / feats_rural.size
    urban_sparsity = 1 - count_ge_threshold(feats_urban, threshold) / feats_urban.size
    print(f"Rural sparsity: {rural_sparsity}, Urban sparsity: {urban_sparsity}")
    feats_all = np.concatenate(feats_all, axis=0)
    sparsity = 1 -count_ge_threshold(feats_all, threshold) / feats_all.size
    print(f"Total sparsity: {sparsity}")
    # save the sparsity score to a file
    with open(os.path.join(args.show_dir, 'sparsity_score.txt'), 'w') as f:
        f.write(f"Rural sparsity: {rural_sparsity}\n")
        f.write(f"Urban sparsity: {urban_sparsity}\n")
        f.write(f"Total sparsity: {sparsity}\n")



if __name__ == '__main__':
    main()
