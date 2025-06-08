# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# ISA Fusion in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(type='SegFormerAdapter', scales=[8]),
    decode_head=dict(
        # loss_decode=dict(
        #     type='CrossEntropyLoss',
        #     use_sigmoid=False,
        #     loss_weight=1.0,
        #     class_weight=[1.25, 1, 1.25, 1.5, 1.25, 1, 1.25],),
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='isa',
                isa_channels=256,
                key_query_num_convs=1,
                down_factor=(8, 8),
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
