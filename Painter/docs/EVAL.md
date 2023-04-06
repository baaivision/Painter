# Evaluation Instructions for Painter

## NYU Depth V2

To evaluate Painter on NYU Depth V2, you may first update the `$JOB_NAME` in `$Painter_ROOT/eval/nyuv2_depth/eval.sh`, then run:
```bash
bash eval/nyuv2_depth/eval.sh
```

## ADE20k Semantic Segmentation

To evaluate Painter on ADE20k semantic segmentation, you may first update the `$JOB_NAME` in `$Painter_ROOT/eval/ade20k_semantic/eval.sh`, then run:
```bash
bash eval/ade20k_semantic/eval.sh
```

## COCO Panoptic Segmentation

To evaluate Painter on COCO panoptic segmentation, you may first update the `$JOB_NAME` in `$Painter_ROOT/eval/coco_panoptic/eval.sh`, then run:
```bash
bash eval/coco_panoptic/eval.sh
```


## COCO Human Pose Estimation

To evaluate Painter on COCO pose estimation, first generate the painted images:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 --use_env eval/mmpose_custom/painter_inference_pose.py --ckpt_path models/painter_vit_large/painter_vit_large.pth
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 --use_env eval/mmpose_custom/painter_inference_pose.py --ckpt_path models/painter_vit_large/painter_vit_large.pth --flip_test
```

Then, you may update the `job_name` and `ckpt_file` in `$Painter_ROOT/eval/mmpose_custom/configs/coco_256x192_test_offline.py`, and run:
```bash
cd $Painter_ROOT/eval/mmpose_custom
./tools/dist_test.sh configs/coco_256x192_test_offline.py none 1 --eval mAP
```

## Low-level Vision Tasks

### Deraining

To evaluate Painter on deraining, first generate the derained images.
```bash
python eval/derain/painter_inference_derain.py --ckpt_path models/painter_vit_large/painter_vit_large.pth
```

Then, update the path to derained images and ground truth in `$Painter_ROOT/eval/derain/evaluate_PSNR_SSIM.m` and run the following script in MATLAB.
```bash
$Painter_ROOT/eval/derain/evaluate_PSNR_SSIM.m 
```


### Denoising

To evaluate Painter on SIDD denoising, first generate the denoised images.
```bash
python eval/sidd/painter_inference_sidd.py --ckpt_path models/painter_vit_large/painter_vit_large.pth
```

Then, update the path to denoising output and ground truth in `$Painter_ROOT/eval/sidd/eval_sidd.m` and run the following script in MATLAB.
```bash
$Painter_ROOT/eval/sidd/eval_sidd.m 
```


### Low-Light Image Enhancement

To evaluate Painter on LoL image enhancement:
```bash
python eval/lol/painter_inference_lol.py --ckpt_path models/painter_vit_large/painter_vit_large.pth
```
