_base_ = './voc_faster_rcnn_energy.py'

custom_imports = dict(
    imports=['projects.DreamBox.dreambox'], allow_failed_imports=False)

backend_args = None
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
data_root = 'data/coco/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='val2017/'),
        ann_file='annotations/instances_val2017_ood_rm_overlap.json',
        pipeline=test_pipeline
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetricWithOOD',
    ann_file=data_root + 'annotations/instances_val2017_ood_rm_overlap.json',
    metric='bbox',
    format_only=True,
    backend_args=backend_args,
    outfile_prefix='work_dirs/voc_faster_rcnn_energy/ood_coco_res')
test_evaluator = val_evaluator