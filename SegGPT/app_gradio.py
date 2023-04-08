# -*- coding: utf-8 -*-

import sys
import io
import requests
import json
import base64
from PIL import Image
import numpy as np
import gradio as gr

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

def resizeImg(img):
    res, hres = 448, 448
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((res, hres))
    temp = io.BytesIO()
    img.save(temp, format="WEBP")
    return base64.b64encode(temp.getvalue()).decode('ascii')

def inference_mask_cat(
              prompt,
              img,
              img_,
              ):
    output_list = [img, img_]
    return output_list


# define app features and run

examples = [
            ['./images/hmbb_1.jpg', './images/hmbb_2.jpg', './images/hmbb_3.jpg'],
            ['./images/rainbow_1.jpg', './images/rainbow_2.jpg', './images/rainbow_3.jpg'],
            ['./images/earth_1.jpg', './images/earth_2.jpg', './images/earth_3.jpg'],
            ['./images/obj_1.jpg', './images/obj_2.jpg', './images/obj_3.jpg'],
            ['./images/xray_1.jpg', './images/xray_2.jpg', './images/xray_3.jpg'],
            ['./images/ydt_2.jpg', './images/ydt_1.jpg', './images/ydt_3.jpg'],
           ]

demo_mask = gr.Interface(fn=inference_mask1,
                   inputs=[gr.ImageMask(brush_radius=8, label="prompt (提示图)"), gr.Image(label="img1 (测试图1)"), gr.Image(label="img2 (测试图2)")],
                    outputs=[gr.Image(label="output1 (输出图1)").style(height=384, width=384), gr.Image(label="output2 (输出图2)").style(height=384, width=384)],
                    #outputs=gr.Gallery(label="outputs (输出图)"),
                    examples=examples,
                    description="<p> \
                    Choose an example below &#128293; &#128293;  &#128293; <br>\
                    Or, upload by yourself: <br>\
                    1. Upload images to be tested to 'img1' and/or 'img2'. <br>2. Upload a prompt image to 'prompt' and draw a mask.  <br>\
                            Tips: The more accurate you annotate, the more accurate the model predicts.;) \
</p>",
                   cache_examples=False,
                   allow_flagging="never",
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

demo = gr.TabbedInterface([demo_mask, ], ['General 1-shot', ], title=title)

#demo.launch(share=True, auth=("baai", "vision"))
demo.launch(enable_queue=False)



