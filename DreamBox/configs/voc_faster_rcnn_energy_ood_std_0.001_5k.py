_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py', 'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.DreamBox.dreambox'], allow_failed_imports=False)

model = dict(
    roi_head=dict(
        type='OODRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxEnergyOODHead',
            num_classes=20,
            loss_ood=dict(
                type='FocalLoss', use_sigmoid=True, loss_weight=10.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerWithOOD'),
            sampler=dict(
                type='RandomSamplerWithOOD')),
    )
)

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotationsWithOOD', with_ood_labels=True),
    dict(type='RandomChoiceResize',
         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                   (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackOODLabelDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# dataset settings
dataset_type = 'CocoDatasetWithOOD'
data_root = 'data/voc/'
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        data_prefix=dict(img='JPEGImages/'),
        ann_file='voc0712_train_all_std_0.001_ood_5k.json'))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='JPEGImages/'),
        ann_file='val_coco_format.json',
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

val_evaluator = dict(
    type='CocoMetricWithOOD',
    ann_file=data_root + 'val_coco_format.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    outfile_prefix='work_dirs/voc_faster_rcnn_energy_ood_std_0.001_5k/id_voc_res')
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=18)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=8,
        by_epoch=True,
        milestones=[12, 16],
        gamma=0.1)
]
