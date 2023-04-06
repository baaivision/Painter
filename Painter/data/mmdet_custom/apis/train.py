# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import random
import sys
import tqdm

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    load_data_only = cfg.custom.get('load_data_only', False)
    assert load_data_only
    train_loader_cfg_custom = copy.deepcopy(train_loader_cfg)
    # train_loader_cfg_custom['shuffle'] = False  # we prefer gen data in order
    # train_loader_cfg_custom['dist'] = False
    data_loaders = [build_dataloader(ds, **train_loader_cfg_custom) for ds in dataset]
    # only enumerate dataset
    for data_loader in data_loaders:
        for _ in tqdm.tqdm(data_loader):
            pass
    print("dataset enumerated, exit!")
    sys.exit()
