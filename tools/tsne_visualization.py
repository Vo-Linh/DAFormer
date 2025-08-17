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

    # Shuffle to avoid plotting sequence artifacts
    indices = np.arange(len(feats))
    np.random.shuffle(indices)
    feats = feats[indices]
    labels = labels[indices]

    # Run t-SNE
    print(f"Running t-SNE on {feats.shape[0]} feature points...")
    time_start = time.time()
    tsne = TSNE(n_jobs=30, n_iter=500, perplexity=30, random_state=seed)  # lower perplexity
    embedded = tsne.fit_transform(feats)
    print(f"t-SNE completed in {time.time() - time_start:.2f} seconds")

    # Visualization
    palette[0] = [0, 0, 0]  # background black
    palette = np.array(palette)
    plt.figure(figsize=(6, 6), dpi=150)

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
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE figure to {out}")
    else:
        plt.show()


def tsne_visualize_2domains(all_feats, all_labels, palette, class_index, class_names, out_path=None, max_samples_per_class=300):
    """
    Visualize 2 domains using t-SNE with proper class separation.
    
    Parameters:
        all_feats: list of list of np.ndarray
            Features for each domain, e.g., [ [feat_batches_domain1], [feat_batches_domain2] ]
        all_labels: list of list of np.ndarray
            Labels for each domain in the same structure as all_feats
        palette: list of [R, G, B]
            Colors for each class
        class_index: list of int
            The class IDs to visualize
        class_names: list of str
            Names for each class
        out_path: str or None
            Path to save the figure
        max_samples_per_class: int
            Maximum samples per class for t-SNE to avoid overcrowding
    """
    
    sampled_feats = []
    sampled_labels = []

    # Iterate over domains
    for domain_idx, (feats, labels) in enumerate(zip(all_feats, all_labels)):
        # Merge batches in each domain
        feats = np.concatenate(feats, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Sample features per class
        for cls_id in class_index:
            cls_mask = (labels == cls_id)
            indices = np.where(cls_mask)[0]
            
            # Randomly sample if too many points
            if len(indices) > max_samples_per_class:
                sampled_idx = np.random.choice(indices, size=max_samples_per_class, replace=False)
            else:
                sampled_idx = indices

            sampled_feats.append(feats[sampled_idx])
            
            # Assign labels uniquely per class (ignoring domain idx for clear class separation)
            sampled_labels.append(np.full(sampled_idx.shape, cls_id))

    # Combine all sampled features & labels
    feats = np.concatenate(sampled_feats, axis=0)
    labels = np.concatenate(sampled_labels, axis=0)

    print(f"Running t-SNE on {feats.shape[0]} feature points...")
    time_start = time.time()
    tsne = TSNE(n_jobs=30, n_iter=500, perplexity=20, random_state=42)
    embedded = tsne.fit_transform(feats)
    print(f"t-SNE completed in {time.time() - time_start:.2f} seconds")

    # Shuffle for better visualization
    indices = np.arange(len(embedded))
    np.random.shuffle(indices)
    embedded = embedded[indices]
    labels = labels[indices]

    # Prepare color palette
    palette = np.array(palette)
    print(f"Using palette for {len(palette)} classes: {palette}")
    print(f"Labels in visualization: {np.unique(labels)}")

    # Plot all points directly (no batching)
    plt.figure(figsize=(6, 6), dpi=150)
    for cls_id in range(len(class_names)):
        mask = labels == cls_id
        if np.any(mask):
            plt.scatter(
                embedded[mask, 0], embedded[mask, 1],
                s=10, alpha=0.7, color=np.array(palette[cls_id]) / 255.0,
                label=class_names[cls_id]
            )

    plt.axis('off')
    # plt.legend(markerscale=2, loc='best', fontsize=10)  # Optional legend

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE figure to {out_path}")
    else:
        plt.show()

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
    
    

    
    feats_all = []
    gts_all = []
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
            feats_all.append(feats_rural)
            gts_all.append(gts_rural)
            out_path = os.path.join(args.show_dir, f'tsne_plot_{domain}.png')
            for i in range(3):
                extract_and_tsne_visualize(feats_rural, gts_rural, dataset.PALETTE, dataset.CLASSES, out_path, seed=i, max_samples=1000)
    
        elif domain == 'Urban':
            feats_urban, gts_urban = single_gpu_infer_features(model, dataloader, args)
            feats_all.append(feats_urban)
            gts_all.append(gts_urban)
            out_path = os.path.join(args.show_dir, f'tsne_plot_{domain}.png')
            for i in range(3):
                extract_and_tsne_visualize(feats_urban, gts_urban, dataset.PALETTE, dataset.CLASSES, out_path, seed=i, max_samples=1000)
    
    
    # for i in range(7):    
    #     out_path = os.path.join(args.show_dir, f'{dataset.CLASSES[i]}_tsne_plot_2domains.png')
    #     tsne_visualize_2domains(feats_all, gts_all, dataset.PALETTE, [ i], ['Rural', 'Urban'], out_path)
    




if __name__ == '__main__':
    main()

# def extract_and_tsne_visualize(all_feats, all_labels, palette, class_names, out_path=None, seed=42):

#     all_feats = np.concatenate(all_feats, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)
#     print(f"Total feature points: {all_feats.shape[0]}, Total labels: {all_labels.shape[0]}")
    

#     sampled_feats = []
#     sampled_labels = []
    
#     for cls_id in np.unique(all_labels):
#         if cls_id >= len(class_names):
#             print(f"Warning: Class ID {cls_id} exceeds class names length {len(class_names)}")
#             continue
#         elif cls_id == 0:
#             print(f"Skipping class ID {cls_id} as it is reserved for background")
#             continue
#         cls_mask = (all_labels == cls_id)
#         indices = np.where(cls_mask)[0]
#         if len(indices) > 300:
#             sampled_idx = np.random.choice(indices, size=300, replace=False)
#         else:
#             sampled_idx = indices
#         sampled_feats.append(all_feats[sampled_idx])
#         sampled_labels.append(np.full_like(sampled_idx, cls_id))

#     feats = np.concatenate(sampled_feats, axis=0)
#     labels = np.concatenate(sampled_labels, axis=0)
        
#     print("Running TSNE on {} feature points...".format(feats.shape[0]))
#     time_start = time.time()
#     tsne = TSNE(n_jobs=30, n_iter=1000, perplexity=50, random_state=seed)
#     embedded = tsne.fit_transform(feats)
#     print(f"t-SNE completed in {time.time() - time_start:.2f} seconds")

#     # Visualization
#     palette[0] = [0, 0, 0]  # Ensure background color is black
#     print(f"Palette: {palette}")
#     palette = np.array(palette)
#     print(f"Labels corresponding to classes: {np.unique(labels)}")
#     print(f"Class names: {class_names}")
#     # plt.figure(figsize=(5, 5))
#     # pro_bar = mmcv.ProgressBar(len(class_names))
#     num_points = len(labels)

#     for cls_id in range(len(class_names)):
#         if cls_id >= len(class_names):
#             print(f"Warning: Class ID {cls_id} exceeds class names length {len(class_names)}")
#             continue
#         idx = labels == cls_id
#         if np.any(idx):
#             plt.scatter(
#                 embedded[idx, 0], embedded[idx, 1],
#                 s=5, alpha=0.7, color=np.array(palette[cls_id]) / 255.0
#             )
#     # for cls_id, cls_name in enumerate(class_names):
#     #     plt.scatter([], [], label=class_names[cls_id], s=5, alpha=0.7, color=np.array(palette[cls_id]) / 255.0)

#     # plt.legend(markerscale=4, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium')
#     # plt.title("Feature Embeddings with t-SNE")
#     # plt.tight_layout()
    
#     if out_path:
#         out = f"{out_path.split('.png')[0]}_{seed}.png"
#         plt.axis('off')
#         plt.savefig(out)
#         print(f"Saved t-SNE figure to {out}")
#     else:
#         plt.show()

# def tsne_visualize_2domains(all_feats, all_labels, palette, class_index, class_names, out_path=None):

#     # feats_1 = np.concatenate(all_feats[0], axis=0)
#     # labels_1 = np.concatenate(all_labels[0], axis=0)
#     # feats_2 = np.concatenate(all_feats[1], axis=0)
#     # labels_2 = np.concatenate(all_labels[1], axis=0)
#     # print(f"Domain 1 - Total feature points: {feats_1.shape[0]}, Total labels: {labels_1.shape[0]}")
#     # print(f"Domain 2 - Total feature points: {feats_2.shape[0]}, Total labels: {labels_2.shape[0]}")    

#     sampled_feats = []
#     sampled_labels = []
#     idx=0
#     for feats, labels in zip(all_feats, all_labels):
#         feats = np.concatenate(feats, axis=0)
#         labels = np.concatenate(labels, axis=0)
#         for cls_id in class_index:
#             cls_mask = (labels == cls_id)
#             indices = np.where(cls_mask)[0]
#             if len(indices) > 300:
#                 sampled_idx = np.random.choice(indices, size=300, replace=False)
#             else:
#                 sampled_idx = indices
#             sampled_feats.append(feats[sampled_idx])
#             sampled_labels.append(np.full_like(sampled_idx, idx))
#         idx += 1
    

#     feats = np.concatenate(sampled_feats, axis=0)
#     labels = np.concatenate(sampled_labels, axis=0)

#     print("Running TSNE on {} feature points...".format(feats.shape[0]))
#     time_start = time.time()
#     tsne = TSNE(n_jobs=30, n_iter=500, perplexity=20, random_state=42)
#     embedded = tsne.fit_transform(feats)
#     print(f"t-SNE completed in {time.time() - time_start:.2f} seconds")
#     indices = np.arange(len(embedded))
#     np.random.shuffle(indices)
    
#     embedded = embedded[indices]
#     labels = labels[indices]

#     # Visualization
#     palette = [[0, 150, 0], [255, 200, 150]]  # Ensure background color is black
#     print(f"Palette: {palette}")
#     palette = np.array(palette)
#     print(f"Labels corresponding to classes: {np.unique(labels)}")
#     print(f"Class names: {class_names}")
#     # plt.figure(figsize=(5,5))
#     # pro_bar = mmcv.ProgressBar(len(class_names))
#     batch_size = 300
#     num_points = len(labels)

#     for i in range((num_points + batch_size - 1) // batch_size):  # Proper range over batches
#         start = i * batch_size
#         end = min((i + 1) * batch_size, num_points)

#         batch_label = labels[start:end]
#         batch_embedded = embedded[start:end]

#         for cls_id in range(len(class_names)):
#             if cls_id >= len(class_names):
#                 print(f"Warning: Class ID {cls_id} exceeds class names length {len(class_names)}")
#                 continue
#             idx = batch_label == cls_id
#             if np.any(idx):
#                 plt.scatter(batch_embedded[idx, 0], batch_embedded[idx, 1],
#                             s=10, color=np.array(palette[cls_id]) / 255.0)
#     # for cls_id, cls_name in enumerate(class_names):
#     #     plt.scatter([], [], label=class_names[cls_id], s=5, alpha=0.7, color=np.array(palette[cls_id]) / 255.0)

#     # plt.legend(markerscale=4, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
#     # plt.title("Feature Embeddings with t-SNE", fontsize=24)
#     # plt.tight_layout()

#     if out_path:
#         plt.savefig(out_path, dpi=300, bbox_inches='tight')
#         print(f"Saved t-SNE figure to {out_path}")
#     else:
#         plt.show()
