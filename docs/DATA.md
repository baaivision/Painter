# Prepare datasets for Painter

The training of our model uses [COCO](https://cocodataset.org/), [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/), [NYUDepthV2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [Synthetic Rain Datasets](https://paperswithcode.com/dataset/synthetic-rain-datasets), [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/), and [LoL](https://daooshee.github.io/BMVC2018website/) datasets.

After processing, the datasets should look like:

```
$Painter_ROOT/datasets/
    nyu_depth_v2/
        sync/
        official_splits/
        nyu_depth_v2_labeled.mat
        datasets/nyu_depth_v2/
        nyuv2_sync_image_depth.json  # generated
        nyuv2_test_image_depth.json  # generated
    ade20k/
        images/
        annotations/
        annotations_detectron2/  # generated
        annotations_with_color/  # generated
        ade20k_training_image_semantic.json  # generated
        ade20k_validation_image_semantic.json  # generated
    ADEChallengeData2016/  # sim-link to $Painter_ROOT/datasets/ade20k
    coco/
        train2017/
        val2017/
        annotations/
            instances_train2017.json
            instances_val2017.json
            person_keypoints_val2017.json
            panoptic_train2017.json
            panoptic_val2017.json
            panoptic_train2017/
            panoptic_val2017/
        panoptic_semseg_val2017/  # generated
        panoptic_val2017/  # sim-link to $Painter_ROOT/datasets/coco/annotations/panoptic_val2017
        pano_sem_seg/  # generated
            panoptic_segm_train2017_with_color
            panoptic_segm_val2017_with_color
            coco_train2017_image_panoptic_sem_seg.json
            coco_val2017_image_panoptic_sem_seg.json
        pano_ca_inst/  # generated
            train_aug0/
            train_aug1/
            ...
            train_aug29/
            train_org/
            train_flip/
            val_org/
            coco_train_image_panoptic_inst.json
            coco_val_image_panoptic_inst.json
    coco_pose/
        person_detection_results/
            COCO_val2017_detections_AP_H_56_person.json
        data_pair/  # generated
            train_256x192_aug0/
            train_256x192_aug1/
            ...
            train_256x192_aug19/
            val_256x192/
            test_256x192/
            test_256x192_flip/
        coco_pose_256x192_train.json  # generated
        coco_pose_256x192_val.json  # generated
    derain/
        train/
            input/
            target/
        test/
            Rain100H/
            Rain100L/
            Test100/
            Test1200/
            Test2800/
        derain_train.json
        derain_test_rain100h.json
    denoise/
        SIDD_Medium_Srgb/
        train/
        val/
        denoise_ssid_train.json  # generated
        denoise_ssid_val.json  # generated
    light_enhance/
        our485/
            low/
            high/
        eval15/
            low/
            high/
        enhance_lol_train.json  # generated
        enhance_lol_val.json  # generated

```
Please follow the following instruction to pre-process individual datasets.


## NYU Depth V2

First, download the dataset from [here](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing). Please make sure to locate the downloaded file to `$Painter_ROOT/datasets/nyu_depth_v2/sync.zip`

Next, prepare [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) test set.
```bash
# get official NYU Depth V2 split file
wget -P datasets/nyu_depth_v2/ http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
# convert mat file to image files
python data/depth/extract_official_train_test_set_from_mat.py datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat data/depth/splits.mat datasets/nyu_depth_v2/official_splits/
```

Lastly, prepare json files for training and evaluation. The generated json files will be saved at `$Painter_ROOT/datasets/nyu_depth_v2/`.
```bash
python data/depth/gen_json_nyuv2_depth.py --split sync
python data/depth/gen_json_nyuv2_depth.py --split test
```

## ADE20k Semantic Segmentation

First, download the dataset from the [official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/), and put it in `$Painter_ROOT/datasets/`. Afterward, unzip the zip file and rename the target folder as `ade20k`. The ADE20k folder should look like:
```
ade20k/
    images/
    annotations/
```

Second, prepare annotations for training using the following command. The generated annotations will be saved at `$Painter_ROOT/datasets/ade20k/annotations_with_color/`.
```bash
python data/ade20k/gen_color_ade20k_sem.py --split training
python data/ade20k/gen_color_ade20k_sem.py --split validation
```

Third, prepare json files for training and evaluation. The generated json files will be saved at `$Painter_ROOT/datasets/ade20k/`.
```bash
python data/ade20k/gen_json_ade20k_sem.py --split training
python data/ade20k/gen_json_ade20k_sem.py --split validation
```

Lastly, to enable evaluation with detectron2, link `$Painter_ROOT/datasets/ade20k` to `$Painter_ROOT/datasets/ADEChallengeData2016` and run:
```bash
# ln -s $Painter_ROOT/datasets/ade20k datasets/ADEChallengeData2016
python data/prepare_ade20k_sem_seg.py
```

## COCO Panoptic Segmentation
Download the COCO2017 dataset and the corresponding panoptic segmentation annotation. The COCO folder should look like:
```
coco/
    train2017/
    val2017/
    annotations/
        instances_train2017.json
        instances_val2017.json
        panoptic_train2017.json
        panoptic_val2017.json
        panoptic_train2017/
        panoptic_val2017/
```

### Prepare Data for COCO Semantic Segmentation
Prepare annotations for training using the following command. The generated annotations will be saved at `$Painter_ROOT/datasets/coco/pano_sem_seg/`.
```bash
python data/coco_semseg/gen_color_coco_panoptic_segm.py --split train2017
python data/coco_semseg/gen_color_coco_panoptic_segm.py --split val2017
```

Prepare json files for training and evaluation. The generated json files will be saved at `$Painter_ROOT/datasets/coco/pano_sem_seg/`.
```bash
python data/coco_semseg/gen_json_coco_panoptic_segm.py --split train2017
python data/coco_semseg/gen_json_coco_panoptic_segm.py --split val2017
```

### Prepare Data for COCO Class-Agnostic Instance Segmentation 

First, pre-process the dataset using the following command, the painted ground truth will be saved to `$Painter_ROOT/datasets/coco/pano_ca_inst`. 

```bash
cd $Painter_ROOT/data/mmdet_custom

# generate training data with common data augmentation for instance segmentation, 
# note we generate 30 copies by alternating train_aug{idx} in configs/coco_panoptic_ca_inst_gen_aug.py
./tools/dist_train.sh configs/coco_panoptic_ca_inst_gen_aug.py 1
# generate training data with only horizontal flip augmentation
./tools/dist_train.sh configs/coco_panoptic_ca_inst_gen_orgflip.py 1
# generate training data w/o data augmentation
./tools/dist_train.sh configs/coco_panoptic_ca_inst_gen_org.py 1

# generate validation data (w/o data augmentation)
./tools/dist_test.sh configs/coco_panoptic_ca_inst_gen_org.py none 1 --eval segm
```

Next, prepare json files for training and evaluation. The generated json files will be saved at `$Painter_ROOT/datasets/coco/pano_ca_inst`.
```bash
cd $Painter_ROOT
python data/mmdet_custom/gen_json_coco_panoptic_inst.py --split train
python data/mmdet_custom/gen_json_coco_panoptic_inst.py --split val
```

Lastly, to enable evaluation with detectron2, link `$Painter_ROOT/datasets/coco/annotations/panoptic_val2017` to `$Painter_ROOT/datasets/coco/panoptic_val2017` and run:
```bash
# ln -s $Painter_ROOT/datasets/coco/annotations/panoptic_val2017 datasets/coco/panoptic_val2017
python data/prepare_coco_semantic_annos_from_panoptic_annos.py
```


## COCO Human Pose Estimation

First, download person detection result of COCO val2017 from [google drive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk), and put it in `$Painter_ROOT/datasets/coco_pose/`


First, pre-process the dataset using the following command, the painted ground truth will be saved to `$Painter_ROOT/datasets/coco_pose/`. 

```bash
cd $Painter_ROOT/data/mmpose_custom

# generate training data with common data augmentation for pose estimation, note we generate 20 copies for training
./tools/dist_train.sh configs/coco_256x192_gendata.py 1
# genearte data for eval during training
./tools/dist_test.sh configs/coco_256x192_gendata.py none 1

# generate data for testing (using offline boxes)
./tools/dist_test.sh configs/coco_256x192_gendata_test.py none 1
# generate data for testing (using offline boxes & with flip)
./tools/dist_test.sh configs/coco_256x192_gendata_testflip.py none 1
```

Next, prepare json files for training and evaluation. The generated json files will be saved at `datasets/pano_ca_inst/`.
```bash
cd $Painter_ROOT
python data/mmpose_custom/gen_json_coco_pose.py --split train
python data/mmpose_custom/gen_json_coco_pose.py --split val
```


## Low-level Vision Tasks

### Deraining
We follow [MPRNet](https://github.com/swz30/MPRNet) to prepare the data for deraining.

Download the dataset following the instructions in [MPRNet](https://github.com/swz30/MPRNet/blob/main/Deraining/Datasets/README.md), and put it in `$Painter_ROOT/datasets/derain/`. The folder should look like:
```bash
derain/
    train/
        input/
        target/
    test/
        Rain100H/
        Rain100L/
        Test100/
        Test1200/
        Test2800/
```

Next, prepare json files for training and evaluation. The generated json files will be saved at `datasets/derain/`.
```bash
python data/derain/gen_json_rain.py --split train
python data/derain/gen_json_rain.py --split val
```

### Denoising
We follow [Uformer](https://github.com/ZhendongWang6/Uformer) to prepare the data for SIDD denoising dataset.

For training data of SIDD, you can download the SIDD-Medium dataset from the [official url](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). For evaluation on SIDD, you can download data from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ev832uKaw2JJhwROKqiXGfMBttyFko_zrDVzfSbFFDoi4Q?e=S3p5hQ).

Next, generate image patches for training by the following command:
```bash
python data/sidd/generate_patches_SIDD.py --src_dir datasets/denoise/SIDD_Medium_Srgb/Data --tar_dir datasets/denoise/train
```

Lastly, prepare json files for training and evaluation. The generated json files will be saved at `datasets/denoise/`.
```bash
python data/sidd/gen_json_sidd.py --split train
python data/sidd/gen_json_sidd.py --split val
```


### Low-Light Image Enhancement

First, download images of LOL dataset from [google drive](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view) and put it in `$Painter_ROOT/datasets/light_enhance/`. The folder should look like:
look like:
```bash
light_enhance/
    our485/
        low/
        high/
    eval15/
        low/
        high/
```

Next, prepare json files for training and evaluation. The generated json files will be saved at `$Painter_ROOTdatasets/light_enhance/`.
```bash
python data/lol/gen_json_lol.py --split train
python data/lol/gen_json_lol.py --split val
```

