# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_vib_mitb5.py',
    '../_base_/datasets/loveda2dalat_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/smda_base.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 111
# Modifications to Basic UDA
smda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.0,
    imnet_feature_dist_classes=[1, 2, 3, 4, 5, 6, 7],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0)
data = dict(
    train=dict(
        # Rare Class Sampling
        # rare_class_sampling=dict(
        #     min_pixels=3000, class_temp=0, min_crop_ratio=0.5))
)) 
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
log_interval = 2000
checkpoint_config = dict(by_epoch=False, interval=log_interval*5, max_keep_ckpts=2)
evaluation = dict(interval=log_interval, metric='mIoU')
# Meta Information for Result Analysis
name = 'DAFormer_LoveDA2DALat_SepASPP_VIBout_MITB5_SMDA'
exp = 'KLTN'
name_dataset = 'rural2urban'
name_architecture = 'daformer_sepaspp_vib_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = ''
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
