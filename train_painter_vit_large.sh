#!/bin/bash

DATA_PATH=datasets
name=painter_vit_large
python -m torch.distributed.launch --nproc_per_node=8 \
	--nnodes=${WORLD_SIZE} --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12358 \
	--use_env main_train.py  \
    --batch_size 2 \
    --accum_iter 16  \
    --model painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 1e-3 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --save_freq 1 \
    --data_path $DATA_PATH/ \
    --json_path  \
    $DATA_PATH/nyu_depth_v2/nyuv2_sync_image_depth.json \
    $DATA_PATH/ade20k/ade20k_training_image_semantic.json \
    $DATA_PATH/coco/pano_ca_inst/coco_train_image_panoptic_inst.json \
    $DATA_PATH/coco/pano_sem_seg/coco_train2017_image_panoptic_sem_seg.json \
    $DATA_PATH/coco_pose/coco_pose_256x192_train.json \
    $DATA_PATH/denoise/denoise_ssid_train.json \
    $DATA_PATH/derain/derain_train.json \
    $DATA_PATH/light_enhance/enhance_lol_train.json \
    --val_json_path \
    $DATA_PATH/nyu_depth_v2/nyuv2_test_image_depth.json \
    $DATA_PATH/ade20k/ade20k_validation_image_semantic.json \
    $DATA_PATH/coco/pano_ca_inst/coco_val_image_panoptic_inst.json \
    $DATA_PATH/coco/pano_sem_seg/coco_val2017_image_panoptic_sem_seg.json \
    $DATA_PATH/coco_pose/coco_pose_256x192_val.json \
    $DATA_PATH/denoise/denoise_ssid_val.json \
    $DATA_PATH/derain/derain_test_rain100h.json \
    $DATA_PATH/light_enhance/enhance_lol_val.json \
    --output_dir models/$name \
    --log_dir models/$name/logs \
    --finetune path/to/mae_pretrain_vit_large.pth \
    # --log_wandb \

