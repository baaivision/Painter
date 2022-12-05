<div align="center">
<h1>Images Speak in Images: <br>A Generalist Painter for In-Context Visual Learning </h1>

[Xinlong Wang](https://www.xloong.wang/)<sup>1*</sup>, &nbsp; [Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl)<sup>1,2*</sup>, &nbsp; [Yue Cao](http://yue-cao.me/)<sup>1*</sup>, &nbsp; [Chunhua Shen](https://cshen.github.io/)<sup>2</sup>, &nbsp; [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1,3</sup>

<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), &nbsp; <sup>2</sup>[ZJU](https://www.zju.edu.cn/english/), &nbsp; <sup>3</sup>[PKU](https://english.pku.edu.cn/)




<br>
  
<image src="teaser.jpg" width="720px" />
<br>

</div>

<br>

We present Painter, a generalist model using an "image"-centric solution for in-context visual learning, that is, to redefine the output of core vision tasks as images, and specify task prompts as also images. With this idea, our training process is extremely simple, which performs standard masked image modeling on the stitch of input and output image pairs. This makes the model capable of performing tasks conditioned on visible image patches. Thus, during inference, we can adopt a pair of input and output images from the same task as the input condition, to indicate which task to perform. Examples of in-context inference are illustrated in the figure above, consisting of seven in-domain examples (seven rows at top) and three out-of-domain examples (three rows at bottom). 

Without bells and whistles, our generalist Painter can achieve competitive performance compared to well-established task-specific models, on seven representative vision tasks ranging from high-level visual understanding to low-level image processing. 
Painter significantly outperforms recent generalist models on several challenging tasks.
Surprisingly, our model shows capabilities of completing out-of-domain tasks, which do not exist in the training data, such as open-category keypoint detection and object segmentation, validating the powerful task transferability of in-context learning. 


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


## Citation

```
@article{Painter,
  title={Images Speak in Images: A Generalist Painter for In-Context Visual Learning},
  author={Wang, Xinlong and Wang, Wen and Cao, Yue and Shen, Chunhua and Huang, Tiejun},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```

## Contact

**We are hiring** at all levels at BAAI Vision Team, including full-time researchers, engineers and interns. 
If you are interested in working with us on **foundation model, visual perception and multimodal learning**, please contact [Xinlong Wang](https://www.xloong.wang/) (`wangxinlong@baai.ac.cn`) and [Yue Cao](http://yue-cao.me/) (`caoyue@baai.ac.cn`).

