backend_args = dict(backend='local')
base_lr = 0.001
codec = dict(
    input_size=(
        224,
        224,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        switch_epoch=390,
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='SetFullImageBBox'),
            dict(type='GetBBoxCenterScale'),
            dict(type='SPNAugmentation'),
            dict(input_size=(
                224,
                224,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    input_size=(
                        224,
                        224,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'cpu_augmentor',
        'tensor_augmentor',
        'bbox',
        'dual_image_loader',
    ])
data_mode = 'topdown'
data_root = '/workspace/speedplusv2/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(interval=10, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = (
    224,
    224,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
max_epochs = 420
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e-3a2dd350.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.375),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                cae_weights_path=
                '/root/RTMPose/satellite/CAE_Weight/model_final.state',
                deepaug_sigma=0.1,
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                prob_deep=0.2,
                prob_identity=0.5,
                prob_randconv=0.2,
                prob_style=0.1,
                randconv_kernel_size=3,
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                type='CombinedAugmentation'),
        ],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                224,
                224,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=384,
        in_featuremap_size=(
            7,
            7,
        ),
        input_size=(
            224,
            224,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=11,
        simcc_split_ratio=2.0,
        type='RTMCCHead'),
    test_cfg=dict(flip_test=False),
    type='TopdownPoseEstimator')
num_keypoints = 11
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.0),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=210,
        begin=210,
        by_epoch=True,
        convert_to_iter_based=True,
        end=420,
        eta_min=5e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=42)
resume = False
stage2_num_epochs = 30
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/validation.json',
        data_mode='topdown',
        data_prefix=dict(img='val/'),
        data_root='/workspace/speedplusv2/',
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='SetFullImageBBox'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                224,
                224,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/workspace/speedplusv2/annotations/validation.json',
    type='CocoMetric')
train_batch_size = 128
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=50)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='annotations/train.json',
        data_mode='topdown',
        data_prefix=dict(img='train/'),
        data_root='/workspace/speedplusv2/',
        metainfo=dict(from_file='configs/_base_/datasets/custom.py'),
        pipeline=[
            dict(
                aux_dir='/root/workspace/speedplusv2/SPIN/',
                type='LoadImageFromDualDir'),
            dict(type='SetFullImageBBox'),
            dict(type='GetBBoxCenterScale'),
            dict(type='SPNAugmentation'),
            dict(input_size=(
                224,
                224,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    input_size=(
                        224,
                        224,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        aux_dir='/root/workspace/speedplusv2/SPIN/',
        type='LoadImageFromDualDir'),
    dict(type='SetFullImageBBox'),
    dict(type='GetBBoxCenterScale'),
    dict(type='SPNAugmentation'),
    dict(input_size=(
        224,
        224,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            input_size=(
                224,
                224,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='SetFullImageBBox'),
    dict(type='GetBBoxCenterScale'),
    dict(type='SPNAugmentation'),
    dict(input_size=(
        224,
        224,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            input_size=(
                224,
                224,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_batch_size = 32
val_cfg = None
val_dataloader = None
val_evaluator = None
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='SetFullImageBBox'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        224,
        224,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs'
