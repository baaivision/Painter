# -*- coding: utf-8 -*-

import sys
import io
import requests
import json
import base64
from PIL import Image
import numpy as np
import gradio as gr

sys.path.append('.')

def inference_mask1_sam(prompt,
              img,
              img_):

    files = {
        "useSam" : 1,
        "pimage" : resizeImg(prompt["image"]),
        "pmask" : resizeImg(prompt["mask"]),
        "img" : resizeImg(img),
        "img_" : resizeImg(img_)
    }
    r = requests.post("http://120.92.79.209/painter/run", json = files)
    a = json.loads(r.text)

    res = []

    for i in range(len(a)):
        #out = Image.open(io.BytesIO(base64.b64decode(a[i])))
        #out = out.resize((224, 224))
        #res.append(np.uint8(np.array(out)))
        res.append(np.uint8(np.array(Image.open(io.BytesIO(base64.b64decode(a[i]))))))

    return res[1:] # remove the prompt image

def inference_mask1(prompt,
              img,
              img_):
    files = {
        "pimage" : resizeImg(prompt["image"]),
        "pmask" : resizeImg(prompt["mask"]),
        "img" : resizeImg(img),
        "img_" : resizeImg(img_)
    }
    r = requests.post("http://120.92.79.209/painter/run", json = files)
    a = json.loads(r.text)
    res = []
    for i in range(len(a)):
        #out = Image.open(io.BytesIO(base64.b64decode(a[i])))
        #out = out.resize((224, 224))
        #res.append(np.uint8(np.array(out)))
        res.append(np.uint8(np.array(Image.open(io.BytesIO(base64.b64decode(a[i]))))))
    return res


def inference_mask_video(
              prompt,
              vid,
              request: gr.Request,
              ):


    files = {
        "pimage" : resizeImgIo(prompt["image"]),
        "pmask" : resizeImgIo(prompt["mask"]),
        "video" : open(vid, 'rb'),
    }
    r = requests.post("http://120.92.79.209/painter/runVideo", files = files)
    '''
    path = str(uuid.uuid4()) + "." + str(time.time())
    fName = 'out.mp4'
    file_out = "video/" + path + "." + fName
    with open(file_out,"wb") as f:
        f.write(r.content)
    '''
    a = json.loads(r.text)
    return [np.uint8(np.array(Image.open(io.BytesIO(base64.b64decode(a["mask"]))))), a["url"]]


def resizeImg(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return base64.b64encode(temp.getvalue()).decode('ascii')

def resizeImgIo(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return io.BytesIO(temp.getvalue())


# define app features and run

examples = [
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg', './images/hmbb_3.jpg'],
            ['./images/rainbow_1.jpg', './images/rainbow_2.jpg', './images/rainbow_3.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg', './images/earth_3.jpg'],
            ['./images/obj_1.jpg', './images/obj_2.jpg', './images/obj_3.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg', './images/ydt_3.jpg'],
           ]

examples_sam = [
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg', './images/hmbb_3.jpg'],
            ['./images/street_1.jpg', './images/street_2.jpg', './images/street_3.jpg'],
            ['./images/tom_1.jpg', './images/tom_2.jpg', './images/tom_3.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg', './images/earth_3.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg', './images/ydt_3.jpg'],
           ]

examples_video = [
            ['./videos/horse-running.jpg', './videos/horse-running.mp4'],
            ['./videos/a_man_is_surfing_3_30.jpg', './videos/a_man_is_surfing_3_30.mp4'],
    ['./videos/a_car_is_moving_on_the_road_40.jpg', './videos/a_car_is_moving_on_the_road_40.mp4'],
['./videos/jeep-moving.jpg', './videos/jeep-moving.mp4'],
['./videos/child-riding_lego.jpg', './videos/child-riding_lego.mp4'],
['./videos/a_man_in_parkour_100.jpg', './videos/a_man_in_parkour_100.mp4'],
]

demo_mask = gr.Interface(fn=inference_mask1,
                   inputs=[gr.ImageMask(brush_radius=8, label="prompt (提示图)"), gr.Image(label="img1 (测试图1)"), gr.Image(label="img2 (测试图2)")],
                    #outputs=[gr.Image(shape=(448, 448), label="output1 (输出图1)"), gr.Image(shape=(448, 448), label="output2 (输出图2)")],
                    outputs=[gr.Image(label="output1 (输出图1)").style(height=256, width=256), gr.Image(label="output2 (输出图2)").style(height=256, width=256)],
                    #outputs=gr.Gallery(label="outputs (输出图)"),
                    examples=examples,
                    #title="SegGPT for Any Segmentation<br>(Painter Inside)",
                    description="<p> \
                    Choose an example below &#128293; &#128293;  &#128293; <br>\
                    Or, upload by yourself: <br>\
                    1. Upload images to be tested to 'img1' and/or 'img2'. <br>2. Upload a prompt image to 'prompt' and draw a mask.  <br>\
                            <br> \
                            💎 The more accurate you annotate, the more accurate the model predicts. <br>\
                            💎 Examples below were never trained and are randomly selected for testing in the wild. <br>\
                            💎 Current UI interface only unleashes a small part of the capabilities of SegGPT, i.e., 1-shot case. \
</p>",
                   cache_examples=False,
                   allow_flagging="never",
                   )

demo_mask_sam = gr.Interface(fn=inference_mask1_sam,
                   inputs=[gr.ImageMask(brush_radius=4, label="prompt (提示图)"), gr.Image(label="img1 (测试图1)"), gr.Image(label="img2 (测试图2)")],
                    #outputs=[gr.Image(shape=(448, 448), label="output1 (输出图1)"), gr.Image(shape=(448, 448), label="output2 (输出图2)")],
                    # outputs=[gr.Image(label="output1 (输出图1)").style(height=256, width=256), gr.Image(label="output2 (输出图2)").style(height=256, width=256)],
                    #outputs=gr.Gallery(label="outputs (输出图)"),
                    outputs=[gr.Image(label="SAM output (mask)").style(height=256, width=256),gr.Image(label="output1 (输出图1)").style(height=256, width=256), gr.Image(label="output2 (输出图2)").style(height=256, width=256)],
                    # outputs=[gr.Image(label="output3 (输出图1)").style(height=256, width=256), gr.Image(label="output4 (输出图2)").style(height=256, width=256)],
                    examples=examples_sam,
                    #title="SegGPT for Any Segmentation<br>(Painter Inside)",
                    description="<p> \
                    <strong>SAM+SegGPT: One touch for segmentation in all images or videos.</strong> <br>\
                    Choose an example below &#128293; &#128293;  &#128293; <br>\
                    Or, upload by yourself: <br>\
                    1. Upload images to be tested to 'img1' and 'img2'. <br>2. Upload a prompt image to 'prompt' and draw <strong>a point or line on the target</strong>.  <br>\
                            <br> \
                            💎 SAM segments the target with any point or scribble, then SegGPT segments all other images. <br>\
                            💎 Examples below were never trained and are randomly selected for testing in the wild. <br>\
                            💎 Current UI interface only unleashes a small part of the capabilities of SegGPT, i.e., 1-shot case. \
</p>",
                   cache_examples=False,
                   allow_flagging="never",
                   )

demo_mask_video = gr.Interface(fn=inference_mask_video,
                   inputs=[gr.ImageMask(label="prompt (提示图)"), gr.Video(label="video (测试视频)").style(height=448, width=448)],
                    outputs=[gr.Image(label="SAM output (mask)").style(height=256, width=256), gr.Video().style(height=448, width=448)],
                    examples=examples_video,
                    description="<p> \
                    <strong>SegGPT+SAM: One touch for any segmentation in a video.</strong> <br>\
                    Choose an example below &#128293; &#128293;  &#128293; <br>\
                    Or, upload by yourself: <br>\
                    1. Upload a video to be tested to 'video'. If failed, please check the codec, we recommend h.264 by default. <br>2. Upload a prompt image to 'prompt' and draw <strong>a point or line on the target</strong>.  <br>\
<br> \
💎 SAM segments the target with any point or scribble, then SegGPT segments the whole video. <br>\
💎 Examples below were never trained and are randomly selected for testing in the wild. <br>\
💎 Current UI interface only unleashes a small part of the capabilities of SegGPT, i.e., 1-shot case. <br> \
                Note: we only take the first 16 frames for the demo.    \
</p>",
                   )



title = "SegGPT: Segmenting Everything In Context<br> \
<div align='center'> \
<h2><a href='https://arxiv.org/abs/2304.03284' target='_blank' rel='noopener'>[paper]</a> \
<a href='https://github.com/baaivision/Painter' target='_blank' rel='noopener'>[code]</a></h2> \
<br> \
<image src='file/rainbow.gif' width='720px' /> \
<h2>SegGPT performs arbitrary segmentation tasks in images or videos via in-context inference, such as object instance, stuff, part, contour, and text, with only one single model.</h2> \
</div> \
"

demo = gr.TabbedInterface([demo_mask_sam, demo_mask_video, demo_mask], ['SAM+SegGPT (一触百通)', '🎬Anything in a Video', 'General 1-shot'], title=title)

#demo.launch(share=True, auth=("baai", "vision"))
demo.launch(enable_queue=False)
#demo.launch(enable_queue=False, server_name="0.0.0.0", server_port=34311)


