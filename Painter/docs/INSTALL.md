# Installation

### Requirements
* Linux, CUDA>=9.2, GCC>=5.4
* PyTorch >= 1.8.1
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### A fix for timm
This repo is based on [timm==0.3.2](https://github.com/huggingface/pytorch-image-models), for which [a fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

---
The installations below are only for data processing and evaluation, but are not required for training.

### Setup for ADE20K Semantic Segmentation

Install [detectron2](https://github.com/facebookresearch/detectron2), following the instructions in [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). 
Or simply use the following command.
```bash
git clone https://github.com/facebookresearch/detectron2
python -m pip install -e detectron2
```

### Setup for COCO Panoptic Segmentation

Install [mmcv](https://github.com/open-mmlab/mmcv), following the instructions in [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). 
Or simply use the following command.
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && MMCV_WITH_OPS=1 pip install -e . -v
```


Install [mmdetection](https://github.com/open-mmlab/mmdetection), following the instructions in [here](https://mmdetection.readthedocs.io/en/stable/get_started.html#installation). 
Or simply use the following command.
<!-- Note we use mmdet @ `e71b4996`. -->
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -v -e .
```


### Setup for COCO Pose Estimation

Install [mmpose](https://github.com/open-mmlab/mmpose) following the instructions in [here](https://mmpose.readthedocs.io/en/v0.29.0/install.html). 
Or simply use the following command.
<!-- * Note we use mmpose @ `8c58a18b` -->
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```


### Setup for Low-Level Vision Tasks

Install MATLAB for evaluation.