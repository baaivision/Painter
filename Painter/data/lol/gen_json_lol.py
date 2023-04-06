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
    parser = argparse.ArgumentParser('Low lignt enhancement preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train', 'val'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/light_enhance')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()

    save_path = os.path.join(args.output_dir, "enhance_lol_{}.json".format(args.split))

    if args.split == 'train':
        image_dir = "datasets/light_enhance/our485/low/"
    elif args.split == 'val':
        image_dir = "datasets/light_enhance/eval15/low/"
    else:
        raise NotImplementedError
    print(save_path)

    output_dict = []

    image_path_list = glob.glob(os.path.join(image_dir, '*.png')) + glob.glob(os.path.join(image_dir, '*.jpg'))
    for image_path in tqdm.tqdm(image_path_list):
        target_path = image_path.replace('low', 'high')
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        pair_dict = {}
        pair_dict["image_path"] = image_path.replace('datasets/', '')
        pair_dict["target_path"] = target_path.replace('datasets/', '')
        pair_dict["type"] = "lol_image2enhance"
        output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
