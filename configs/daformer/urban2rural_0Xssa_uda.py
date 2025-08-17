# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/resnetv1c_disciminator_optimizer_loveda.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/ssda_urban_Xrural_to_rural_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/ass_uda.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed

DATA_ROOT = '/home/Hung_Data/HungData/mmseg_data/Datasets'
seed = 1

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4) 
# Optimizer Hyperparameters
optimizer_config = None

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=103000)
# Logging Configuration

checkpoint_config = dict(by_epoch=False, interval=1000,max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

# Meta Information for Result Analysis
name = 'exp2_urban5rural_2rural_ASS_UDA'

exp = 'ASS_UDA'
name_dataset = 'urban2rural'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'resnetv1c'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
