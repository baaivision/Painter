<div align="center">
<h1>Images Speak in Images: <br>A Generalist Painter for In-Context Visual Learning </h1>

[Xinlong Wang](https://www.xloong.wang/)<sup>1*</sup>, &nbsp; [Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl)<sup>1,2*</sup>, &nbsp; [Yue Cao](http://yue-cao.me/)<sup>1*</sup>, &nbsp; [Chunhua Shen](https://cshen.github.io/)<sup>2</sup>, &nbsp; [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1,3</sup>

<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), &nbsp; <sup>2</sup>[ZJU](https://www.zju.edu.cn/english/), &nbsp; <sup>3</sup>[PKU](https://english.pku.edu.cn/)

CVPR 2023


<br>
  
<image src="docs/teaser.jpg" width="720px" />
<br>

</div>

<br>

We present Painter, a generalist model using an "image"-centric solution for in-context visual learning, that is, to redefine the output of core vision tasks as images, and specify task prompts as also images. With this idea, our training process is extremely simple, which performs standard masked image modeling on the stitch of input and output image pairs. This makes the model capable of performing tasks conditioned on visible image patches. Thus, during inference, we can adopt a pair of input and output images from the same task as the input condition, to indicate which task to perform. Examples of in-context inference are illustrated in the figure above, consisting of seven in-domain examples (seven rows at top) and three out-of-domain examples (three rows at bottom).
Without bells and whistles, our generalist Painter can achieve competitive performance compared to well-established task-specific models, on seven representative vision tasks ranging from high-level visual understanding to low-level image processing. 
In addition, Painter significantly outperforms recent generalist models on several challenging tasks.

[[Paper]](https://arxiv.org/abs/2212.02499)

## Hightlights

### $\color{#2F6EBA}{Images\ Speak\ in\ Images}$ 

- image as the general-purpose interface
- redefine the output spaces of vision tasks as images

### $\color{#2F6EBA}{A\ Generalist\ Painter}$ 

- given an input image, prediction is to inpaint the desired but missing output "image"
- excellent performance on 7 representative vision tasks with a single generalist model

### $\color{#2F6EBA}{In{-}Context\ Visual\ Learning}$  
- automatically perform vision tasks according to the input task prompts 
- even the tasks do not exist in the training data


## Installation
See [installation instructions](docs/INSTALL.md).

## Data
See [data instructions](docs/DATA.md). 

We also provide [a toy training dataset](https://huggingface.co/BAAI/Painter/blob/main/toy_datasets.tar), with 10 samples from each required datasets. You can put it in `$Painter_ROOT/toy_datasets` and set `DATA_PATH=toy_datasets` in `$Painter_ROOT/train_painter_vit_large.sh` for toy experiments.

## Training
Download pre-trained MAE ViT-Large model from [here](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) and update `path/to/mae_pretrain_vit_large.pth` in `$Painter_ROOT/train_painter_vit_large.sh`. 

We use 8 nodes (<code>total_bsz = 8x8x32 = 2048</code>) for training:



```bash
bash train_painter_vit_large.sh
```

## Evaluation
See [evaluation instructions](docs/EVAL.md). 

A pre-trained Painter is available at [🤗 Hugging Face Models](https://huggingface.co/BAAI/Painter/blob/main/painter_vit_large.pth). The results on various tasks are summarized below:


<table border="1" width="100%">
	<tr align="center">
        <!-- <th> Task </th>  -->
        <th colspan="3"> depth estimation </th><th colspan="1"> semantic seg. </th><th colspan="1">panoptic seg.</th><th colspan="1">keypoint det.</th> <th colspan="2"> denoising </th> <th colspan="2"> deraining </th> <th colspan="2"> enhance.</th>
    </tr>
  	<tr align="center">
        <!-- <th> Dataset </th>  -->
        <th colspan="3"> NYU v2 </th><th colspan="1"> ADE20k </th><th colspan="1"> COCO 2017 </th><th colspan="1"> COCO 2017 </th> <th colspan="2"> SIDD </th> <th colspan="2"> 5 datasets </th> <th colspan="2"> LoL </th>
    </tr>
  <tr align="center">
        <!-- <th> Metric </th>  -->
        <th> RMSE </th> <th> A.Rel </th> <th> d1 </th> <th colspan="1"> mIoU </th><th colspan="1">PQ</th><th colspan="1">AP</th> <th> PSNR </th> <th> SSIM </th>  <th> PSNR </th> <th> SSIM </th>  <th> PSNR </th> <th> SSIM </th>
    </tr>
    <tr align="center">
        <!-- <th> Painter </th>  -->
        <th> 0.288 </th> <th> 0.080 </th> <th> 0.950 </th> <th colspan="1"> 49.9 </th> <th> 43.4 </th> <th>72.1</th> <th>  38.66 </th> <th> 0.954 </th>  <th> 29.42 </th> <th> 0.867 </th>  <th> 22.34 </th> <th> 0.872 </th>
    </tr>
</table>
<br>


## Citation

```
@article{Painter,
  title={Images Speak in Images: A Generalist Painter for In-Context Visual Learning},
  author={Wang, Xinlong and Wang, Wen and Cao, Yue and Shen, Chunhua and Huang, Tiejun},
  journal={arXiv preprint arXiv:2212.02499},
  year={2022}
}
```

## Acknowledgement
[MAE](https://github.com/facebookresearch/mae), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [detectron2](https://github.com/facebookresearch/detectron2), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [bts](https://github.com/cleinc/bts), [mmcv](https://github.com/open-mmlab/mmcv), [mmdetetection](https://github.com/open-mmlab/mmdetection), [mmpose](https://github.com/open-mmlab/mmpose), [MIRNet](https://github.com/swz30/MIRNet), [MPRNet](https://github.com/swz30/MPRNet), and [Uformer](https://github.com/ZhendongWang6/Uformer).

## Contact

**We are hiring** at all levels at BAAI Vision Team, including full-time researchers, engineers and interns. 
If you are interested in working with us on **foundation model, visual perception and multimodal learning**, please contact [Xinlong Wang](https://www.xloong.wang/) (`wangxinlong@baai.ac.cn`) and [Yue Cao](http://yue-cao.me/) (`caoyue@baai.ac.cn`).

