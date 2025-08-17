norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DLV2Head',
        in_channels=2048,
        in_index=3,
        dilations=(6, 12, 18, 24),
        num_classes=7,
        align_corners=False,
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

discriminator_g=dict(
    type='GlobalDiscriminator',
    in_channels=7,  # same as num_classes
    base_channels=64,
)

discriminator_s=dict(
    type='SemanticDiscriminatorCSA',
    in_channels=2048,  # feature map channels (last ResNet stage)
    num_classes=7   # 2 * num_classes
)

gan_loss=dict(
    type='GANLoss',
    gan_type='vanilla',
    loss_weight=1.0
)

lambda_seg=1.0,
lambda_gadv=0.01,
lambda_sadv=0.05,
lambda_gd=1.0,
lambda_sd=1.0,
ignore_index=255



optimizer = dict(
    model=dict(type='Adam', lr=2.5e-4, betas=(0.9, 0.999), weight_decay=0.0005),
    discriminator_g=dict(type='Adam', lr=1e-4, betas=(0.5, 0.999)),
    discriminator_s=dict(type='Adam', lr=1e-4, betas=(0.5, 0.999))
)
optimizer_config = dict()