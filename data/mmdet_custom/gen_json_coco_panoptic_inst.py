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
import json
import tqdm
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('COCO class-agnostic instance segmentation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train', 'val'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/coco/pano_ca_inst')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()

    panoptic_dir = "datasets/coco/pano_ca_inst"
    save_path = os.path.join(args.output_dir, "coco_{}_image_panoptic_inst.json".format(args.split))
    print(save_path)

    output_dict = []

    image_path_list = glob.glob(os.path.join(panoptic_dir, '{}_*'.format(args.split), '*image*.png'))
    for image_path in tqdm.tqdm(image_path_list):
        image_dir, image_name = os.path.dirname(image_path), os.path.basename(image_path)
        panoptic_path = os.path.join(image_dir, image_name.replace('image', 'label'))
        assert os.path.isfile(image_path)
        if not os.path.isfile(panoptic_path):
            print("ignore {}".format(image_path))
            continue
        pair_dict = {}
        pair_dict["image_path"] = image_path.replace('datasets/', '')
        pair_dict["target_path"] = panoptic_path.replace('datasets/', '')
        pair_dict["type"] = "coco_image2panoptic_inst"
        output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
