# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_isa_bottleneck_mitb5.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/smda_u100r2R_640x640.py',
    # Basic UDA Self-Training
    '../_base_/uda/smda_base.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 11
# Modifications to Basic UDA
smda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.00,
    imnet_feature_dist_classes=[1, 2, 3, 4, 5, 6, 7],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120)
# Adjusted min_pixels for Rare Class Sampling from 3000 to 10000
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=10000, class_temp=0.01, min_crop_ratio=0.5))) 
#
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=2)
evaluation = dict(interval=2000, metric='mIoU')
# Meta Information for Result Analysis
name = 'u100r2R_smda_daformer_isa_640x640'
exp = 'basic_smda'
name_dataset = 'urban2rural'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
