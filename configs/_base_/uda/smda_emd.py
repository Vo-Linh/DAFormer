# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='TrustAwareSMDACS',
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    debug_gt_interval = 10,
    sm_prob=0.1,
    print_grad_magnitude=False,
    cmap='loveda',
    trust_update_interval=100,
    # coeffient=0.5  # Coefficient for trust weight adjustment
    # trust_update_interval=500,
    
)
use_ddp_wrapper = True
