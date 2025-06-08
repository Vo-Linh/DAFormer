# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/supervised_daformer_sepaspp_mitb5.py',
    
    '../_base_/datasets/urban_to_rural_512x512.py',
    # # Basic UDA Self-Training
    # '../_base_/uda/smda_base.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10.py'
]
# Random Seed
seed = 111

    
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
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=10)
evaluation = dict(interval=log_interval, metric='mIoU')
# Meta Information for Result Analysis
name = 'DAFormer_Supervised_Rural+Urban'
exp = 'KLTN'
name_dataset = ''
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = ''
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
