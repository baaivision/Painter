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
    parser = argparse.ArgumentParser('NYU Depth V2 preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['sync', 'test'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/nyu_depth_v2')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    split2dir = {
        'sync': 'sync',
        'test': 'official_splits/test',
    }

    split_dir = split2dir[args.split]
    output_dict = []
    save_path = os.path.join(args.output_dir, "nyuv2_{}_image_depth.json".format(args.split))

    src_dir = os.path.join("datasets/nyu_depth_v2", split_dir)
    image_path_list = glob.glob(src_dir + "/*/rgb_*.jpg")

    for image_path in tqdm.tqdm(image_path_list):
        room_name = image_path.split('/')[-2]
        frame_name = image_path.split('/')[-1].split('.')[0].split('_')[1]
        target_path = src_dir + '/' + room_name + '/sync_depth_' + frame_name + '.png'
        assert os.path.isfile(image_path)
        assert os.path.isfile(target_path)
        image_name = image_path.split('{}/'.format(args.split))[-1]
        target_name = target_path.split('{}/'.format(args.split))[-1]

        pair_dict = {}
        pair_dict["image_path"] = "nyu_depth_v2/{}/".format(split_dir) + image_name
        pair_dict["target_path"] = "nyu_depth_v2/{}/".format(split_dir) + target_name
        pair_dict["type"] = "nyuv2_image2depth"
        output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
