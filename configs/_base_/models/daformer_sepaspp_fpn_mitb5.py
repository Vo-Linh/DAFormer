# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# DAFormer with VIB Bottleneck
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/home/Hung_Data/HungData/HaiDang/PiPa/pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=128,
        out_channel_=[64, 128, 320, 512],
        num_outs=4),
    auxiliary_head=dict(
        type='FPN_VIB_Head',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=7,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='KLLoss', loss_weight=0.1,)),  
    decode_head=dict(
        type='DAFormerVIBHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                # _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg)
            ),
        vib_params=None,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ),
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
