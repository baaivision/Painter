# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import copy
import os
import argparse
import glob
import json
import warnings
import tqdm
import sys
sys.path.insert(0, "data")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from .gen_color_coco_stuff_sem import PALETTE
# from .gen_color_ade20k_sem import unique, colorEncode

from skimage.segmentation import find_boundaries
from panopticapi.utils import rgb2id, IdGenerator


# define colors according to mean separation
def define_colors_by_mean_sep(num_colors=133, channelsep=7):
    num_sep_per_channel = channelsep
    separation_per_channel = 256 // num_sep_per_channel

    color_dict = {}
    # R = G = B = 0
    # B += separation_per_channel  # offset for the first loop
    for location in range(num_colors):
        num_seq_r = location // num_sep_per_channel ** 2
        num_seq_g = (location % num_sep_per_channel ** 2) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_r <= num_sep_per_channel) and (num_seq_g <= num_sep_per_channel) \
               and (num_seq_b <= num_sep_per_channel)

        R = 255 - num_seq_r * separation_per_channel
        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        assert (R < 256) and (G < 256) and (B < 256)
        assert (R >= 0) and (G >= 0) and (B >= 0)
        assert (R, G, B) not in color_dict.values()

        color_dict[location] = (R, G, B)
        # print(location, (num_seq_r, num_seq_g, num_seq_b), (R, G, B))
    return color_dict


def load_image_with_retry(image_path):
    while True:
        try:
            img = Image.open(image_path)
            return img
        except OSError as e:
            print(f"Catched exception: {str(e)}. Re-trying...")
            import time
            time.sleep(1)


def get_args_parser():
    parser = argparse.ArgumentParser('COCO panoptic segmentation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train2017', 'val2017'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/coco/pano_sem_seg')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()
    split = args.split
    channelsep = 7

    json_file = 'datasets/coco/annotations/panoptic_{}.json'.format(split)
    segmentations_folder = 'datasets/coco/annotations/panoptic_{}'.format(split)
    img_folder = 'datasets/coco/{}'.format(split)
    panoptic_coco_categories = 'data/panoptic_coco_categories.json'
    output_dir = os.path.join(args.output_dir, 'panoptic_segm_{}_with_color'.format(split))
    print(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        warnings.warn("{} exists! make sure to overwrite?".format(output_dir))
        # raise NotImplementedError("{} exists! make sure to overwrite?".format(output_dir))

    # load cat info
    with open(panoptic_coco_categories, 'r') as f:
        categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}
    catid2colorid = {category['id']: idx for idx, category in enumerate(categories_list)}

    # define colors (dict of cat_id to color mapper)
    num_colors = len(categories)
    color_dict = define_colors_by_mean_sep(num_colors=num_colors, channelsep=channelsep)

    # load image annos
    with open(json_file, 'r') as f:
        coco_d = json.load(f)

    num_iscrowd = 0
    for ann in tqdm.tqdm(coco_d['annotations']):
        # save the time for loading images
        # # find input img that correspond to the annotation
        segmentation_org = np.array(
            load_image_with_retry(os.path.join(segmentations_folder, ann['file_name'])),
            dtype=np.uint8
        )
        segmentation_id = rgb2id(segmentation_org)

        image_height_segm, image_width_segm = segmentation_org.shape[0], segmentation_org.shape[1]
        image_height, image_width = image_height_segm, image_width_segm

        segmentation = copy.deepcopy(segmentation_org)
        segmentation[:, :, :] = 0

        boxes = [seg['bbox'] for seg in ann['segments_info']]  # x, y, w, h
        if len(boxes) == 0:
            print("bbox is empty!")
            continue
        boxes = np.array(boxes)  # (num_boxes, 4)

        for segment_info in ann['segments_info']:
            # retrieval color using class id
            catid = segment_info['category_id']
            colorid = catid2colorid[catid]
            color = color_dict[colorid]
            # paint color
            mask = segmentation_id == segment_info['id']
            segmentation[mask] = color

        segmentation = Image.fromarray(segmentation)
        segmentation.save(os.path.join(output_dir, ann['file_name']))
