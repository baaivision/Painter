# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.lr_decay as lrd
import util.misc as misc
from util.misc import get_parameter_groups
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

import models_painter

from engine_train import train_one_epoch, evaluate_pt

from data.pairdataset import PairDataset
import data.pair_transforms as pair_transforms
from util.masking_generator import MaskingGenerator
from data.sampler import DistributedSamplerWrapper

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def get_args_parser():
    parser = argparse.ArgumentParser('Painter pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, nargs='+',
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--num_mask_patches', default=784, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    parser.add_argument('--stop_grad_patch_embed', action='store_true',
                        help='stop-grad after first conv, or patch embedding')
    parser.set_defaults(stop_grad_patch_embed=False)
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--drop_path', default=0., type=float,
                        help='Drop path rate (default: 0.)')
    parser.add_argument('--min_random_scale', default=0.3, type=float,
                        help='Minimal random scale for randomresizecrop (default: 0.3)')
    parser.add_argument('--last_norm_instance', action='store_true', default=False,
                    help='use instance norm to normalize each channel map before the decoder layer')
    parser.add_argument('--half_mask_ratio', default=0.1, type=float,
                        help='ratio of using half mask during training (default: 0.1)')
    parser.add_argument('--use_checkpoint', action='store_true', default=False,
                    help='use checkpoint to save GPU memory')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save checkkpoints frequency')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--layer_decay', type=float, default=1.0, metavar='LRD',
                        help='Learning rate layer decay')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    #parser.add_argument('--json_path', default='./', type=str,
    parser.add_argument('--json_path', default='./', nargs='+', type=str,
                        help='json path')
    parser.add_argument('--val_json_path', default='./', nargs='+',type=str,
                        help='json path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--use_two_pairs', action='store_true',
                        help='concatenate two pairs of images')
    parser.set_defaults(use_two_pairs=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--enable_deepspeed',
                        action='store_true', default=False)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')
	# misc
    parser.add_argument('--log_wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    misc.init_distributed_mode(args)

    if ds_init is not None:
        misc.create_ds_config(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # define the model
    model = models_painter.__dict__[args.model]()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        rm_key_list = ['decoder_embed.weight', 'decoder_embed.bias',  'mask_token']
        if args.last_norm_instance:
            rm_key_list.extend(['norm.weight', 'norm.bias'])
        for k in rm_key_list:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate patch embedding
        if "patch32" in args.model:
            patch_weight = checkpoint['model']['patch_embed.proj.weight']
            new_patch_weight = torch.nn.functional.interpolate(patch_weight, size=(32, 32), mode='bicubic', align_corners=False)
            checkpoint['model']['patch_embed.proj.weight'] = new_patch_weight

        # interpolate position embedding
        if "painter" not in args.model:
            interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size[0] // patch_size, args.input_size[1] // patch_size)
    args.patch_size = patch_size

    # simple augmentation
    transform_train = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size[1], scale=(args.min_random_scale, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.RandomApply([
                pair_transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            pair_transforms.RandomHorizontalFlip(),
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train2 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train3 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train_seccrop = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size, scale=(args.min_random_scale, 1.0), ratio=(0.3, 0.7), interpolation=3),  # 3 is bicubic
            ])
    transform_val = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    masked_position_generator = MaskingGenerator(
        args.window_size, num_masking_patches=args.num_mask_patches,
        max_num_patches=args.max_mask_patches_per_block,
        min_num_patches=args.min_mask_patches_per_block,
    )
    dataset_train = PairDataset(args.data_path, args.json_path, transform=transform_train, transform2=transform_train2, transform3=transform_train3, transform_seccrop=transform_train_seccrop, masked_position_generator=masked_position_generator, use_two_pairs=args.use_two_pairs, half_mask_ratio=args.half_mask_ratio)
    dataset_val = PairDataset(args.data_path, args.val_json_path, transform=transform_val, transform2=None, transform3=None, masked_position_generator=masked_position_generator, use_two_pairs=args.use_two_pairs, half_mask_ratio=1.0)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        num_samples_train = len(dataset_train)
        weights_train = dataset_train.weights
        sampler_train = torch.utils.data.WeightedRandomSampler(weights_train, num_samples_train, replacement=True)
        sampler_train = DistributedSamplerWrapper(sampler_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.log_wandb:
        experiment = args.log_dir.split('/')[-2]
        if args.resume == '':
            wandb.init(project="Painter", name=experiment, config=args)
        else:
            wandb.init(project="Painter", name=experiment, config=args, resume=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        )
    

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, model.no_weight_decay()
        )
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" %
              model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.accum_iter
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # following timm: set wd as 0 for bias and norm layers
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=args.opt_betas)
        print(optimizer)
        loss_scaler = NativeScaler()

    misc.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            global_rank=global_rank,
            args=args
        )
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate_pt(data_loader_val, model, device, epoch=epoch, global_rank=global_rank, args=args)
        print(f"Val loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if global_rank == 0 and args.log_wandb:
        wandb.finish()


if __name__ == '__main__':
    args, ds_init = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, ds_init)
