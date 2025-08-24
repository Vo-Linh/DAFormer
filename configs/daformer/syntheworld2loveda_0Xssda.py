# Obtained from: https://github.com/lhoyer/DAFormer
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv2_r50-d8_discriminator_optimizer.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/ssda_syntheworld_Xloveda_to_loveda_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/assda.py',
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
runner = dict(type='IterBasedRunner', max_iters=50000)
# Logging Configuration

checkpoint_config = dict(by_epoch=False, interval=1000,max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

# Meta Information for Result Analysis
name = 'exp2_syntheworld5loveda_2loveda_assda_Aug_18'

exp = 'AssDA'
name_dataset = 'urban2rural'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'deeplabv2'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
