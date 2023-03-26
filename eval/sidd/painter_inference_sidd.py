# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import sys
import os
import warnings

import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy.io as sio

sys.path.append('.')
import models_painter

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SIDD denoising', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='9_9')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--save', action='store_true', help='save predictions',
                        default=False)
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1'):
    # build model
    model = getattr(models_painter, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, tgt, size, model, out_path, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)

    valid = torch.ones_like(tgt)
    loss, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = output * imagenet_std + imagenet_mean
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]

    return output.numpy()


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_path = args.ckpt_path
    model = args.model
    prompt = args.prompt
    input_size = args.input_size

    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]
    dst_dir = os.path.join('models_inference', ckpt_dir.split('/')[-1],
                           "sidd_inference_{}_{}".format(ckpt_file, os.path.basename(prompt).split(".")[0]))
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    print("output_dir: {}".format(dst_dir))

    model_painter = prepare_model(ckpt_path, model)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    img_src_dir = "datasets/denoise/val/"
    filepath = os.path.join(img_src_dir, 'ValidationNoisyBlocksSrgb.mat')
    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))  # (40, 32, 256, 256, 3)
    Inoisy /= 255.

    img2_path = "datasets/denoise/train/input/{}.png".format(prompt)
    tgt2_path = "datasets/denoise/train/groundtruth/{}.png".format(prompt)

    # load the shared prompt image pair
    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize((input_size, input_size))
    img2 = np.array(img2) / 255.

    tgt2 = Image.open(tgt2_path)
    tgt2 = tgt2.resize((input_size, input_size))
    tgt2 = np.array(tgt2) / 255.

    model_painter.eval()
    restored = np.zeros_like(Inoisy)
    for img_idx in tqdm.tqdm(range(40)):
        for patch_idx in range(32):
            """ Load an image """
            img_org = Inoisy[img_idx, patch_idx, :, :, :]

            img = cv2.resize(img_org, (input_size, input_size))
            # img = img_org.resize((input_size, input_size))

            img = np.concatenate((img2, img), axis=0)
            assert img.shape == (input_size * 2, input_size, 3)
            # normalize by ImageNet mean and std
            img = img - imagenet_mean
            img = img / imagenet_std

            tgt = tgt2  # tgt is not available
            tgt = np.concatenate((tgt2, tgt), axis=0)

            assert tgt.shape == (input_size * 2, input_size, 3)
            # normalize by ImageNet mean and std
            tgt = tgt - imagenet_mean
            tgt = tgt / imagenet_std

            # make random mask reproducible (comment out to make it change)
            torch.manual_seed(2)

            output = run_one_image(img, tgt, size=(256, 256), model=model_painter, out_path=None, device=device)
            rgb_restored = output
            rgb_restored = np.clip(rgb_restored, 0, 1)
            restored[img_idx, patch_idx, :, :, :] = rgb_restored

            # optionally save images
            if args.save:
                out_path = os.path.join(dst_dir, '%04d_%02d.png' % (img_idx + 1, patch_idx + 1))
                output = rgb_restored * 255
                output = Image.fromarray(output.astype(np.uint8))
                output.save(out_path)

    # save denoised data
    sio.savemat(os.path.join(dst_dir, 'Idenoised.mat'), {"Idenoised": restored, })
    print(os.path.join(dst_dir, 'Idenoised.mat'))
