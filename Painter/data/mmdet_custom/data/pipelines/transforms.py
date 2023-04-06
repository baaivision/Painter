# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import os
from PIL import Image
import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.datasets.builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


def define_colors_per_location_r_gb(num_location_r=16, num_location_gb=20):
    sep_r = 255 // num_location_r
    sep_gb = 256 // num_location_gb + 1  # +1 for bigger sep in gb

    color_dict = {}
    # R = G = B = 0
    # B += separation_per_channel  # offset for the first loop
    for global_y in range(4):
        for global_x in range(4):
            global_locat = (global_x, global_y)
            global_locat_sum = global_y * 4 + global_x
            R = 255 - global_locat_sum * sep_r
            for local_y in range(num_location_gb):
                for local_x in range(num_location_gb):
                    local_locat = (local_x, local_y)
                    G = 255 - local_y * sep_gb
                    B = 255 - local_x * sep_gb

                    assert (R < 256) and (G < 256) and (B < 256)
                    assert (R >= 0) and (G >= 0) and (B >= 0)
                    assert (R, G, B) not in color_dict.values()

                    location = (global_locat, local_locat)
                    color_dict[location] = (R, G, B)

    # colors = [v for k, v in color_dict.items()]
    return color_dict


def simplify_color_dict(color_dict, num_location_r=16, num_location_gb=20):
    color_dict_simple = {}
    for k, v in color_dict.items():
        global_locat, local_locat = k
        global_x, global_y = global_locat
        local_x, local_y = local_locat
        absolute_x = global_x * num_location_gb + local_x
        absolute_y = global_y * num_location_gb + local_y
        color_dict_simple[(absolute_x, absolute_y)] = np.array(v)
    return color_dict_simple


@PIPELINES.register_module()
class SaveDataPairCustom:
    """Save PanoInst Masks

    """

    def __init__(self,
                 dir_name,
                 target_path='../datasets/coco/pano_ca_inst',
                 method='mass_center',
                 num_location_r=16,
                 num_location_gb=20):
        self.dir_name = dir_name
        self.target_path = target_path
        output_dir = os.path.join(self.target_path, self.dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.method = method
        self.color_dict_global_local = define_colors_per_location_r_gb(
            num_location_r=num_location_r, num_location_gb=num_location_gb)
        self.color_dict = simplify_color_dict(
            self.color_dict_global_local, num_location_r=num_location_r, num_location_gb=num_location_gb)

    def __call__(self, results):
        """Call function to save images.
        """
        # get keys of interest
        img = results['img']  # (h, w, 3), ndarray, range 0-255
        gt_bboxes = results['gt_bboxes']  # (num_inst, 4), ndarray, xyxy
        gt_labels = results['gt_labels']  # (num_inst, )
        gt_masks = results['gt_masks'].masks  # BitmapMasks, gt_masks.masks: (num_inst, h, w)
        # gt_semantic_seg = results['gt_semantic_seg']
        # check input
        assert (gt_labels >= 0).all() and (gt_labels < 80).all()
        assert (np.sum(gt_masks, axis=0) >= 0).all() and (np.sum(gt_masks, axis=0) <= 1).all()
        # get box centers
        h, w, _ = img.shape
        num_inst = len(gt_labels)
        segmentation = np.zeros((h, w, 3), dtype="uint8")
        for idx in range(num_inst):
            # iscrowd already filtered, and are stored in results['ann_info']['bboxes_ignore']
            # but some iscrowd are not correctly labelled, e.g., 000000415447
            # if (np.sum(gt_bboxes[idx] == results['ann_info']['bboxes_ignore'], axis=1) == 4).any():
            # if len(results['ann_info']['bboxes_ignore']) > 0:
            #     import pdb; pdb.set_trace()
            if self.method == "geo_center":
                box = gt_bboxes[idx]  # (4, )
                center = (box[:2] + box[2:]) / 2  # (2, )
                center_x, center_y = center
            elif self.method == "mass_center":
                mask = gt_masks[idx]  # (h, w)
                center_x, center_y = self.center_of_mass(mask)
            else:
                raise NotImplementedError(self.method)
            center_x_norm = int(center_x / w * 79)
            center_y_norm = int(center_y / h * 79)
            color = self.color_dict[(center_x_norm, center_y_norm)]
            mask = gt_masks[idx].astype("bool")  # only bool can be used for slicing!
            segmentation[mask] = color
        if (segmentation == 0).all():
            # pure black label
            return results
        # save files
        output_dir = os.path.join(self.target_path, self.dir_name)
        file_name = results['img_info']['file_name']
        # images are loaded in bgr order, reverse before saving
        img_pil = Image.fromarray(img[:, :, ::-1].astype('uint8'))
        label_pil = Image.fromarray(segmentation)
        image_path = os.path.join(output_dir, file_name.replace(".jpg", "_image_{}.png".format(self.dir_name)))
        label_path = os.path.join(output_dir, file_name.replace(".jpg", "_label_{}.png".format(self.dir_name)))
        # if os.path.exists(image_path) or os.path.exists(label_path):
        #     print("{} exists!".format(image_path))
        #     return results
        aug_idx = 0
        while os.path.exists(image_path) or os.path.exists(label_path):
            aug_idx += 1
            image_path = os.path.join(output_dir, file_name.replace(".jpg", "_image_{}_{}.png".format(self.dir_name, aug_idx)))
            label_path = os.path.join(output_dir, file_name.replace(".jpg", "_label_{}_{}.png".format(self.dir_name, aug_idx)))
        img_pil.save(image_path)
        label_pil.save(label_path)

        return results

    def center_of_mass(self, mask, esp=1e-6):
        """Calculate the centroid coordinates of the mask.

        Args:
            mask (Tensor): The mask to be calculated, shape (h, w).
            esp (float): Avoid dividing by zero. Default: 1e-6.

        Returns:
            tuple[Tensor]: the coordinates of the center point of the mask.

                - center_h (Tensor): the center point of the height.
                - center_w (Tensor): the center point of the width.
        """
        h, w = mask.shape
        grid_h = np.arange(h)[:, None]
        grid_w = np.arange(w)
        normalizer = mask.sum().astype("float").clip(min=esp)
        center_h = (mask * grid_h).sum() / normalizer
        center_w = (mask * grid_w).sum() / normalizer
        return center_w, center_h

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(method={self.method})'
        return repr_str
