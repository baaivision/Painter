import os
import argparse

import torch
import numpy as np

from seggpt_engine import inference_image, inference_video
import models_seggpt

from seggpt_inference import prepare_model

from PIL import Image

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def main():
    input_image = 'input/images/IMG_0124.JPG'
    save_masks = True

    frame_prompts = {
        'masks': [
            "input/masks/IMG_0084_frame_mask.jpg",
        "input/masks/IMG_0100_frame_mask.jpg"
        ],
        'images': [
            "input/images/IMG_0113.JPG",
            "input/images/IMG_0126.JPG"
        ]
    }


    capped_prompts = {
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

    # run inference to get frame
    prompt_images = frame_prompts['images']
    prompt_masks = frame_prompts['masks']
    frame_output = os.path.join(output_path, 'frame.png')
    frame_overlay_output = os.path.join(output_path, 'frame_overlay.png')
    frame_mask = inference_image(model, device, input_image, prompt_images, prompt_masks, frame_output, frame_overlay_output, return_mask=True, upscale=True)

    # now convert frame_mask to an image to see it
    # threshold mask
    threshold = 20
    frame_mask = frame_mask.max(axis=-1)  # convert to greyscale
    frame_mask = frame_mask > threshold
    if save_masks:
        frame_mask_img = Image.fromarray(255 * frame_mask.astype(np.uint8), mode='L')
        frame_mask_img.save(os.path.join(output_path, 'frame_mask.png'))

    # run inference to get capped cells
    prompt_images = capped_prompts['images']
    prompt_masks = capped_prompts['masks']
    capped_output = os.path.join(output_path, 'capped.png')
    capped_overlay_output = os.path.join(output_path, 'capped_overlay.png')
    capped_mask = inference_image(model, device, input_image, prompt_images, prompt_masks, capped_output, capped_overlay_output, return_mask=True, upscale=True)

    # convert capped_mask to an image to see it
    threshold = 20
    capped_mask = capped_mask.max(axis=-1)  # convert to greyscale
    capped_mask = (capped_mask > threshold) & frame_mask
    if save_masks:
        capped_mask_img = Image.fromarray(255 * capped_mask.astype(np.uint8), mode='L')
        capped_mask_img.save(os.path.join(output_path, 'capped_mask.png'))


    # determine number of pixels that make up frame
    frame_pixels = np.count_nonzero(frame_mask)  # could also just do sum
    print("Number of pixels for the frame:", frame_pixels)

    # determine number of pixels that make up capped cells
    capped_pixels = np.count_nonzero(capped_mask)
    print("Number of pixels for capped cells:", capped_pixels)

    # calculate percentage of capped cells
    capped_fraction = capped_pixels / frame_pixels

    print("Capped fraction:", capped_fraction)

    # TODO: Only use capped cells segmented in frame area for improved
    # accuracy --> could we add this as an extra prompt?
    # TODO: Don't scale back up to original resolution, not necessary for
    # fraction

    print(f"Finished!")



if __name__ == '__main__':
    main()