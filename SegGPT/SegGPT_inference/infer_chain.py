import os
import argparse

import torch
import numpy as np

from seggpt_engine import inference_image, inference_video
import models_seggpt

from seggpt_inference import prepare_model

from PIL import Image

import time

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def main():
    input_image = 'input/images/IMG_0124.JPG'
    save_masks = True

    area_prompts = {
        'masks': [
            "input/masks/IMG_0084_frame_mask.jpg",
        "input/masks/IMG_0100_frame_mask.jpg"
        ],
        'images': [
            "input/images/IMG_0113.JPG",
            "input/images/IMG_0126.JPG"
        ]
    }


    object_prompts = {
        'masks': [
            "input/masks/IMG_0113_capped_mask.jpg",
            "input/masks/IMG_0126_capped_mask.jpg" 
        ],
        'images': [
            "input/images/IMG_0113.JPG",
            "input/images/IMG_0126.JPG"
        ]
    }

    output_path = 'chain-output'
    os.makedirs(output_path, exist_ok=True)

    # prepare model
    device = 'cuda'
    default_ckpt_path = "seggpt_vit_large.pth"
    default_model = "seggpt_vit_large_patch16_input896x448"
    default_seg_type = "instance"  # TODO: Try semantic
    model = prepare_model(default_ckpt_path, default_model, default_seg_type).to(device)

    # run inference to get area
    prompt_images = area_prompts['images']
    prompt_masks = area_prompts['masks']
    area_output = os.path.join(output_path, 'area.png')
    area_overlay_output = os.path.join(output_path, 'area_overlay.png')
    area_mask = inference_image(model, device, input_image, prompt_images, prompt_masks, area_output, area_overlay_output, return_mask=True, upscale=False)

    # now convert area_mask to an image to see it
    # threshold mask
    threshold = 20
    area_mask = area_mask.max(axis=-1)  # convert to greyscale
    area_mask = area_mask > threshold
    if save_masks:
        area_mask_img = Image.fromarray(255 * area_mask.astype(np.uint8), mode='L')
        area_mask_img.save(os.path.join(output_path, 'area_mask.png'))

    # run inference to get objects
    prompt_images = object_prompts['images']
    prompt_masks = object_prompts['masks']
    object_output = os.path.join(output_path, 'object.png')
    object_overlay_output = os.path.join(output_path, 'object_overlay.png')
    object_mask = inference_image(model, device, input_image, prompt_images, prompt_masks, object_output, object_overlay_output, return_mask=True, upscale=False)

    # convert object_mask to an image to see it
    threshold = 20
    object_mask = object_mask.max(axis=-1)  # convert to greyscale
    object_mask = (object_mask > threshold) & area_mask
    if save_masks:
        object_mask_img = Image.fromarray(255 * object_mask.astype(np.uint8), mode='L')
        object_mask_img.save(os.path.join(output_path, 'object_mask.png'))


    # determine number of pixels that make up area
    area_pixels = np.count_nonzero(area_mask)  # could also just do sum
    print("Number of pixels for the area:", area_pixels)

    # determine number of pixels that make up object cells
    object_pixels = np.count_nonzero(object_mask)
    print("Number of pixels for objects:", object_pixels)

    # calculate percentage of objects
    object_fraction = object_pixels / area_pixels

    print("object fraction:", object_fraction)

    # TODO: Only use object cells segmented in area area for improved
    # accuracy --> could we add this as an extra prompt?
    # TODO: Don't scale back up to original resolution, not necessary for
    # fraction

    print(f"Finished!")



if __name__ == '__main__':
    st = time.process_time()
    wt = time.perf_counter()
    main()

    et = time.process_time()
    ewt = time.perf_counter()

    print(f'Finished in {et-st} seconds, wall time {ewt- wt}')