# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import numpy as np
import wandb


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm
    # return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    global_rank=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    wandb_images = []
    for data_iter_step, (samples, targets, bool_masked_pos, valid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, targets, bool_masked_pos=bool_masked_pos, valid=valid)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= accum_iter
            model.backward(loss)
            model.step()

            # if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
            # grad_norm = None
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            loss /= accum_iter
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                    parameters=model.parameters(),
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(grad_norm=grad_norm)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            if global_rank == 0 and args.log_wandb:
                wandb.log({'train_loss': loss_value_reduce, 'lr': lr, 'train_loss_scale': loss_scale_value, 'grad_norm': grad_norm})
                if len(wandb_images) < 20:
                    imagenet_mean = np.array([0.485, 0.456, 0.406])
                    imagenet_std = np.array([0.229, 0.224, 0.225]) 
                    y = y[[0]]
                    y = model.module.unpatchify(y)
                    y = torch.einsum('nchw->nhwc', y).detach().cpu()
                    mask = mask[[0]]
                    mask = mask.detach().float().cpu()
                    mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size**2 *3)  # (N, H*W, p*p*3)
                    mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
                    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
                    x = samples[[0]]
                    x = x.detach().float().cpu()
                    x = torch.einsum('nchw->nhwc', x)
                    tgt = targets[[0]]
                    tgt = tgt.detach().float().cpu()
                    tgt = torch.einsum('nchw->nhwc', tgt)
                    im_masked = tgt * (1 - mask)
                    
                    frame = torch.cat((x, im_masked, y, tgt), dim=2)
                    frame = frame[0]
                    frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).int()
                    wandb_images.append(wandb.Image(frame.numpy(), caption="x; im_masked; y; tgt"))

    if global_rank == 0 and args.log_wandb and len(wandb_images) > 0:
        wandb.log({"Training examples": wandb_images})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_pt(data_loader, model, device, epoch=None, global_rank=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    wandb_images = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[1]
        bool_masked_pos = batch[2]
        valid = batch[3]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, targets, bool_masked_pos=bool_masked_pos, valid=valid)

        metric_logger.update(loss=loss.item())
        if global_rank == 0 and args.log_wandb:
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225]) 
            y = y[[0]]
            y = model.module.unpatchify(y)
            y = torch.einsum('nchw->nhwc', y).detach().cpu()
            mask = mask[[0]]
            mask = mask.detach().float().cpu()
            mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size**2 *3)  # (N, H*W, p*p*3)
            mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
            x = samples[[0]]
            x = x.detach().float().cpu()
            x = torch.einsum('nchw->nhwc', x)
            tgt = targets[[0]]
            tgt = tgt.detach().float().cpu()
            tgt = torch.einsum('nchw->nhwc', tgt)
            im_masked = tgt * (1 - mask)
            
            frame = torch.cat((x, im_masked, y, tgt), dim=2)
            frame = frame[0]
            frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).int()
            wandb_images.append(wandb.Image(frame.numpy(), caption="x; im_masked; y; tgt"))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Val loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if global_rank == 0 and args.log_wandb:
        wandb.log({**{f'test_{k}': v for k, v in out.items()},'epoch': epoch})
        if len(wandb_images) > 0:
            wandb.log({"Testing examples": wandb_images[::2][:20]})
    return out
