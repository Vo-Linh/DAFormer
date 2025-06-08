# Obtained from: https://github.com/lhoyer/HRDA
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'LoveDADataset'
data_root = '/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA/loveDA_rural/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512,512) # follow base line of DAFormer
loveda_scale=(1024,1024)
source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=loveda_scale),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=loveda_scale),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg',]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=loveda_scale,
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


train_tartget_dataset = dict(
    type='SMLoveDADataset',
    data_root='/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA/Val/Urban',
    img_dir='images_png',
    ann_dir='masks_png',
    split="/home/Hung_Data/HungData/Thien/DAFormer/configs/splits/ValUrban_10percent.txt",
    pipeline=target_train_pipeline,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='SMDADataset',
        source=dict(
            type='LoveDADataset',
            data_root='/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA/loveDA_rural/',
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            pipeline=source_train_pipeline),
        target=train_tartget_dataset,
    ),
    val=dict(
        type='LoveDADataset',
        data_root='/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA/Train/Urban',
            img_dir='images_png',
            ann_dir='masks_png',
        pipeline=test_pipeline),
    test=dict(
        type='LoveDADataset',
        data_root='/home/Hung_Data/HungData/mmseg_data/Datasets/LoveDA/loveDA/Test/Urban',
        img_dir='images_png',
        ann_dir='masks_png',
        pipeline=test_pipeline))