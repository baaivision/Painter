import os
import argparse

import torch
import numpy as np

from seggpt_engine import inference_image, inference_video
import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt-path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input-image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--input-video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num-frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt-image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt-target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg-type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output-dir', type=str, help='path to output folder for output mask',
                        default='./')
    parser.add_argument('--overlay-dir', type=str, help='path to output folder for combined mask and input (for visualising)', default=None)
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    assert args.input_image or args.input_video and not (args.input_image and args.input_video)
    if args.input_image is not None:
        assert args.prompt_image is not None and args.prompt_target is not None

        img_name = os.path.basename(args.input_image)
        out_path = os.path.join(args.output_dir,'.'.join(img_name.split('.')[:-1]) + '.png')
        if args.overlay_dir is not None:
            ovl_path = os.path.join(args.overlay_dir, '.'.join(img_name.split('.')[:-1]) + '.png')
        else:
            ovl_path = None

        inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path, ovl_path)
    
    if args.input_video is not None:
        assert args.prompt_target is not None and len(args.prompt_target) == 1
        vid_name = os.path.basename(args.input_video)
        out_path = os.path.join(args.output_dir, '.'.join(vid_name.split('.')[:-1]) + '.mp4')
        if args.overlay_dir is not None:
            ovl_path = os.path.join(args.overlay_dir, '.'.join(img_name.split('.')[:-1]) + '.mp4')
        else:
            ovl_path = None

        inference_video(model, device, args.input_video, args.num_frames, args.prompt_image, args.prompt_target, out_path, ovl_path)

    print('Finished.')
