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
    parser = argparse.ArgumentParser('COCO pose estimation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train', 'val'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/coco_pose')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    split = args.split

    if split == "train":
        aug_list = [
            "_aug0", "_aug1", "_aug2", "_aug3", "_aug4",
            "_aug5", "_aug6", "_aug7", "_aug8", "_aug9",
            "_aug10", "_aug11", "_aug12", "_aug13", "_aug14",
            "_aug15", "_aug16", "_aug17", "_aug18", "_aug19",
        ]
    elif split == "val":
        aug_list = ["", "_flip"]
    else:
        raise NotImplementedError

    save_path = os.path.join(args.output_dir, "coco_pose_256x192_{}.json".format(split))
    print(save_path)

    output_dict = []

    for aug_idx in aug_list:
        image_dir = "datasets/coco_pose/data_pair/{}_256x192{}".format(split, aug_idx)
        print(aug_idx, image_dir)
        image_path_list = glob.glob(os.path.join(image_dir, '*image.png'))

        for image_path in tqdm.tqdm(image_path_list):
            label_path = image_path.replace("image.png", "label.png")
            assert label_path != image_path
            assert os.path.isfile(image_path)
            if not os.path.isfile(label_path):
                print("ignoring {}".format(label_path))
                continue
            pair_dict = {}
            pair_dict["image_path"] = image_path.replace('datasets/', '')
            pair_dict["target_path"] = label_path.replace('datasets/', '')
            pair_dict["type"] = "coco_image2pose"
            output_dict.append(pair_dict)

    json.dump(output_dict, open(save_path, 'w'))
