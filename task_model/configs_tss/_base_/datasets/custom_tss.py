# dataset settings
dataset_type = 'CocoCarDataset'
# data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]



# To be modified when this code is applied on other environment
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_c_num50_with_pruning_step1.json', # 32k images
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_c_num50_step1_modified.json', # modified step1
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/random_8000_coco.json', # rondom 8000 coco
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/41k_images.json', # 41k images
        #ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_c_matchcars.json',
        # ann_file='/home/himanshu/Desktop/SnP_region100_vehicle_8000_random_c_keyu_coco_only.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/task_model/mmdetection/SnP_region100_vehicle_8000_random_c_num50_last4_datasets_himanshu.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_c_num50_mmd_vit_l14_336px_mmd_modified.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_c_num100_mmd_vit_l14_336px_mmd_modified_without_sampling_v1.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/task_model/mmdetection/SnP_region100_vehicle_8000_random_c_num100_last4_datasets_frompkl_100vit.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_c_num75_mmd_vit_l14_336px_mmd_modified_pruning_really_exclude_testa.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_c_num75_mmd_vit_l14_336px_mmd_modified_pruning_hdbscan.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_mmd_vit_l14_336px_mmd_pruning_hdbscan_minclustersize15_umap_ncomp10.json',
        # ann_file = '/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_mmd_vit_l14_336px_mmd_pruning_hdbscan_minclustersize15_umap_ncomp10_minclustersize500.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_mmd_vit_l14_336px_mmd_pruning_hdbscan_minclustersize300_umap_ncomp10_manhattan.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_mmd_vit_l14_336px_mmd_pruning_hdbscan_minclustersize300_umap_ncomp10_cosine.json',
        # ann_file='/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/SnP_region100_vehicle_8000_random_mmd_vit_l14_336px_mmd_pruning_hdbscan_minclustersize1000_umap_ncomp10_consine.json',
        # ann_file = '/home/himanshu/comp/code/DataCV2024-main/CCDR_detection/json/SnP_region100_vehicle_8000_random_c_num50.json',
        ann_file='/home/himanshu/comp/code/DataCV2024-main/SnP_detection/json/random_8000_coco.json',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file='/home/himanshu/coco_validation_annots.json',
        ann_file='/home/himanshu/testA_annotation_car_change_path.json',
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/home/himanshu/comp/data/region_100/testA_image_list.json',
        img_prefix='/home/himanshu/comp/data/region_100/testA/',
        pipeline=test_pipeline))
        
        
evaluation = dict(interval=6, metric='bbox')
