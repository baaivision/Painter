# modified from mask2former config
_base_ = [
    './_base_/dataset/coco_panoptic.py', './_base_/default_runtime.py'
]
num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = None

# dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', flip_ratio=1.0),
    # # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(1.0, 1.0),
        multiscale_mode='range',
        keep_ratio=False),
    # dict(
    #     type='RandomCrop',
    #     crop_size=image_size,
    #     crop_type='absolute',
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(
        type='SaveDataPairCustom',
        dir_name='train_orgflip',
        target_path='../../datasets/coco/pano_ca_inst',
    ),  # custom, we don't care the transforms afterward
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(1.0, 1.0),
        multiscale_mode='range',
        keep_ratio=False),
    dict(type='Pad', size=image_size),
    dict(
        type='SaveDataPairCustom',
        dir_name='val_org',
        target_path='../../datasets/coco/pano_ca_inst',
    ),  # custom, we don't care the transforms afterward
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

data_root = '../../datasets/coco/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline,
        ins_ann_file=data_root + 'annotations/instances_val2017.json',
    ),
    test=dict(
        pipeline=test_pipeline,
        ins_ann_file=data_root + 'annotations/instances_val2017.json',
    ))

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

custom = dict(
    load_data_only=True,
)
by_epoch = True
# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=by_epoch,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=by_epoch,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

max_iters = 368750
# runner = dict(type='IterBasedRunner', max_iters=max_iters)
runner = dict(type='EpochBasedRunner', max_epochs=1)  # we prefer by epoch

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=by_epoch),
        dict(type='TensorboardLoggerHook', by_epoch=by_epoch)
    ])
interval = 5000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=by_epoch, interval=interval, save_last=True, max_keep_ckpts=3)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
evaluation = dict(
    interval=interval,
    dynamic_intervals=dynamic_intervals,
    metric=['PQ', 'bbox', 'segm'])

# import newly registered module
custom_imports = dict(
    imports=[
        'data.coco_panoptic',
        'data.pipelines.transforms',
    ],
    allow_failed_imports=False)
