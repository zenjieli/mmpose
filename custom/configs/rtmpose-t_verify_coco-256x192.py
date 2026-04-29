"""Quick verification config: RTMPose-t on full COCO person keypoints.

Runs 10 epochs on real COCO train2017 / val2017 to confirm the model learns.
Uses the official pretrained CSPNeXt backbone so AP improves quickly.
Single-GPU friendly (batch_size=64). Albumentation removed for simplicity.

Usage:
    python tools/train.py custom/configs/rtmpose-t_verify_coco-256x192.py
"""
_base_ = ['../../configs/_base_/default_runtime.py']

# ── runtime ───────────────────────────────────────────────────────────────────
max_epochs = 10
# Official config uses base_lr=4e-3 with effective batch 1024+ (8 GPUs × 256).
# Scale linearly for single-GPU batch_size=64: 4e-3 × (64/1024) = 2.5e-4
base_lr = 2.5e-4

train_cfg = dict(max_epochs=max_epochs, val_interval=2)
randomness = dict(seed=21)

# ── optimizer ─────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# ── lr schedule ───────────────────────────────────────────────────────────────
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# LR is already set for batch_size=64; auto_scale_lr not used
# auto_scale_lr = dict(base_batch_size=1024)

# ── codec ─────────────────────────────────────────────────────────────────────
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# ── model ─────────────────────────────────────────────────────────────────────
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='BN'),   # BN instead of SyncBN for single GPU
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=384,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=(6, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# ── dataset ───────────────────────────────────────────────────────────────────
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

backend_args = dict(backend='local')

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        pipeline=val_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

# ── hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/AP',
        rule='greater',
        interval=2))
