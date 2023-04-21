## **SegGPT Usage**
- We release the [SegGPT model](https://huggingface.co/BAAI/SegGPT/blob/main/seggpt_vit_large.pth) and inference code for segmentation everything, as well as some example images and videos.
### Installation
```
git clone https://github.com/baaivision/Painter
cd Painter/SegGPT/SegGPT_inference && wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth
pip install -r requirements.txt
```
### Usage
Everything in an image with a prompt.
```
python seggpt_inference.py \
--input_image examples/hmbb_2.jpg \
--prompt_image examples/hmbb_1.jpg \
--prompt_target examples/hmbb_1_target.png \
--output_dir ./
```

Everything in an image with multiple prompts.
```
python seggpt_inference.py \
--input_image examples/hmbb_3.jpg \
--prompt_image examples/hmbb_1.jpg examples/hmbb_2.jpg \
--prompt_target examples/hmbb_1_target.png examples/hmbb_2_target.png \
--output_dir ./
```

Everything in a video using a prompt image.
```
python seggpt_inference.py \
--input_video examples/video_1.mp4 \
--prompt_image examples/video_1.jpg \
--prompt_target examples/video_1_target.png \
--output_dir ./
```

Everything in a video using the first frame as the prompt.
```
python seggpt_inference.py \
--input_video examples/video_1.mp4 \
--prompt_target examples/video_1_target.png \
--output_dir ./
```

Processing a long video with prompts from both a target image and the predictions of the previous NUM_FRAMES frames.
```
NUM_FRAMES=4
python seggpt_inference.py \
--input_video examples/video_3.mp4 \
--prompt_target examples/video_3_target.png \
--num_frames $NUM_FRAMES \
--output_dir ./
```

<!-- <div align="center">
<image src="rainbow.gif" width="720px" />
</div> -->
