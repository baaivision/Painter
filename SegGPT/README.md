<div align="center">
<h1>SegGPT: Segmenting Everything In Context </h1>

[Xinlong Wang](https://www.xloong.wang/)<sup>1*</sup>, &nbsp; [Xiaosong Zhang](https://scholar.google.com/citations?user=98exn6wAAAAJ&hl=en)<sup>1*</sup>, &nbsp; [Yue Cao](http://yue-cao.me/)<sup>1*</sup>, &nbsp; [Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl)<sup>2</sup>, &nbsp;  [Chunhua Shen](https://cshen.github.io/)<sup>2</sup>, &nbsp; [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1,3</sup>

<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), &nbsp; <sup>2</sup>[ZJU](https://www.zju.edu.cn/english/), &nbsp; <sup>3</sup>[PKU](https://english.pku.edu.cn/)

Enjoy the [Demo](https://huggingface.co/spaces/BAAI/SegGPT)


<br>
  
<image src="seggpt_teaser.png" width="720px" />
<br>

</div>

<br>

   We present SegGPT, a generalist model for segmenting everything in context. With only one single model, SegGPT can perform arbitrary segmentation tasks in images or videos via in-context inference, such as object instance, stuff, part, contour, and text. 
   SegGPT is evaluated on a broad range of tasks, including few-shot semantic segmentation, video object segmentation, semantic segmentation, and panoptic segmentation. 
   Our results show strong capabilities in segmenting in-domain and out-of-domain targets, either qualitatively or quantitatively. 

[[Paper]](https://arxiv.org/abs/2304.03284)
[[Demo]](https://huggingface.co/spaces/BAAI/SegGPT)

## **Run the demo**
- We provide a UI  with gradio for running the demo locally. Running the following command in a terminal will launch the demo: 
    ```
    python app_gradio.py
    ```
- This demo is also hosted on HuggingFace [here](https://huggingface.co/spaces/BAAI/SegGPT).
- The current UI interface just unleashes a small part of the capabilities of SegGPT. Please stay tuned for more demonstrations.

<div align="center">
<image src="rainbow.gif" width="720px" />
</div>


## Citation

```
@article{SegGPT,
  title={SegGPT: Segmenting Everything In Context},
  author={Wang, Xinlong and Zhang, Xiaosong and Cao, Yue and Wang, Wen and Shen, Chunhua and Huang, Tiejun},
  journal={arXiv preprint arXiv:2304.03284},
  year={2023}
}
```

## Contact

**We are hiring** at all levels at BAAI Vision Team, including full-time researchers, engineers and interns. 
If you are interested in working with us on **foundation model, visual perception and multimodal learning**, please contact [Xinlong Wang](https://www.xloong.wang/) (`wangxinlong@baai.ac.cn`) and [Yue Cao](http://yue-cao.me/) (`caoyue@baai.ac.cn`).

