# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os
import glob
import argparse
import json
import tqdm
import sys
sys.path.insert(0, "data")

import numpy as np
from PIL import Image


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    "copied from https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/utils.py"
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    "Modified from https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/utils.py"
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in unique(labelmap):
        if label <= 0:
            continue
        # note the color_index = class_index - 1
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(np.array(colors[label-1], dtype=np.uint8), (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def define_colors_per_location_mean_sep():
    num_locations = 150
    num_sep_per_channel = int(num_locations ** (1 / 3)) + 1  # 19
    separation_per_channel = 256 // num_sep_per_channel

    color_list = []
    for location in range(num_locations):
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
        assert (R, G, B) not in color_list

        color_list.append((R, G, B))
        # print(location, (num_seq_r, num_seq_g, num_seq_b), (R, G, B))

    return color_list


PALETTE = define_colors_per_location_mean_sep()


def get_args_parser():
    parser = argparse.ArgumentParser('ADE20k semantic segmentation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['training', 'validation'], required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    image_dir = os.path.join("datasets/ade20k/images", args.split)
    segm_dir = os.path.join("datasets/ade20k/annotations", args.split)
    save_dir = os.path.join("datasets/ade20k/annotations_with_color", args.split)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    color_list = define_colors_per_location_mean_sep()

    segm_path_list = glob.glob(os.path.join(segm_dir, '*.png'))
    for segm_path in tqdm.tqdm(segm_path_list):
        # check files
        file_name = os.path.basename(segm_path)
        # in ade20k, images are jpegs, while segms are pngs
        image_path = os.path.join(image_dir, file_name.replace('.png', '.jpg'))
        assert os.path.isfile(segm_path)
        assert os.path.isfile(image_path)

        # paint colors on segm
        segm = Image.open(segm_path)
        segm_color = colorEncode(labelmap=np.array(segm), colors=color_list).astype(np.uint8)
        segm_color = Image.fromarray(segm_color)
        segm_color.save(os.path.join(save_dir, file_name))
