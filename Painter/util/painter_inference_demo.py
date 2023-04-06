# -*- coding: utf-8 -*-

import sys
import os
import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
import models_painter


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('Painter Demo Inference', add_help=False)
    parser.add_argument('--ckpt_dir', type=str, help='dir to ckpt',
                        default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--epoch', type=int, help='model epochs',
                        default=14)
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1'):
    # build model
    model = getattr(models_painter, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.eval()
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
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='nearest').permute(0, 2, 3, 1)[0]
    output = output.int()
    output = Image.fromarray(output.numpy().astype(np.uint8))
    output.save(out_path)


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_dir = args.ckpt_dir
    model = args.model
    epoch = args.epoch

    ckpt_file = 'checkpoint-{}.pth'.format(epoch)
    assert ckpt_dir[-1] != "/"

    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    model_painter = prepare_model(ckpt_path, model)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    img2_path = "path/to/img2"
    tgt2_path = "path/to/tgt2"
    img_path = "path/to/img"
    img_name = os.path.basename(img_path)
    out_path = os.path.join("path/to/out", img_name.replace('.jpg', '.png'))

    res = 448

    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize((res, res))
    img2 = np.array(img2) / 255.

    tgt2 = Image.open(tgt2_path).convert("RGB")
    tgt2 = tgt2.resize((res, res), Image.NEAREST)
    tgt2 = np.array(tgt2) / 255.

    img = Image.open(img_path).convert("RGB")
    size = img.size
    img = img.resize((res, res))
    img = np.array(img) / 255.

    tgt = tgt2  # tgt is not available
    tgt = np.concatenate((tgt2, tgt), axis=0)

    img = np.concatenate((img2, img), axis=0)
    assert img.shape == (2*res, res, 3)
    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std


    assert tgt.shape == (2*res, res, 3)
    # normalize by ImageNet mean and std
    tgt = tgt - imagenet_mean
    tgt = tgt / imagenet_std

    torch.manual_seed(2)
    run_one_image(img, tgt, size, model_painter, out_path, device)

