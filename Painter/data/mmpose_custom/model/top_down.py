# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import mmcv
import numpy as np
from PIL import Image
import torch
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from mmpose.models import builder
from mmpose.models.builder import POSENETS
# from .base import BasePose
from mmpose.models.detectors import TopDown

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16

from mmpose.core.post_processing import flip_back
from data.pipelines.custom_transform import define_colors_gb_mean_sep
color_dict = define_colors_gb_mean_sep()
color_list = [v for k, v in color_dict.items()]
color_list.append((0, 0))


@POSENETS.register_module()
class TopDownCustom(TopDown):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """
    colors = torch.tensor(color_list, dtype=torch.float32, device="cuda")

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            keypoint_head=keypoint_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            loss_pose=loss_pose)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                pseudo_test=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C (Default: 3)
            - img height: imgH
            - img width: imgW
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses. \
                Otherwise, return predicted poses, boxes, image paths \
                and heatmaps.
        """
        if pseudo_test:
            return self.forward_pseudo_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = img.flip(3)  # (b, c, h, w)
            features_flipped = self.backbone(img_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                output_heatmap = (output_heatmap + output_flipped_heatmap)
                if self.test_cfg.get('regression_flip_shift', False):
                    output_heatmap[..., 0] -= 1.0 / img_width
                output_heatmap = output_heatmap / 2

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def forward_pseudo_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        output_heatmap = self.decode_images_to_heatmaps_minmax(
            images=img, resize=False,
        )

        # add support for flip test
        if self.test_cfg.get('flip_test', True):
            image_flip_list = []
            for batch_idx in range(img.shape[0]):
                flip_image_dir = os.path.dirname(img_metas[batch_idx]['image_file']) + "_flip"
                flip_image_name = os.path.basename(img_metas[batch_idx]['image_file'])
                flip_image_path = os.path.join(flip_image_dir, flip_image_name)
                image = np.array(Image.open(flip_image_path))
                image_tensor = torch.from_numpy(image).to(img.device)
                image_flip_list.append(image_tensor)
            img_flipped = torch.stack(image_flip_list)  # (b, h, w, 3)
            if self.with_keypoint:
                # output_flipped_heatmap = self.keypoint_head.inference_model(
                #     features_flipped, img_metas[0]['flip_pairs'])
                output = self.decode_images_to_heatmaps_minmax(
                    images=img_flipped, resize=False,
                )
                flip_pairs = img_metas[0]['flip_pairs']
                assert flip_pairs is not None
                output_flipped_heatmap = flip_back(
                    output,
                    flip_pairs,
                    target_type=self.keypoint_head.target_type)
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if self.test_cfg.get('shift_heatmap', False):
                    output_flipped_heatmap[:, :, :, 1:] = output_flipped_heatmap[:, :, :, :-1]
                output_heatmap = (output_heatmap + output_flipped_heatmap)
                if self.test_cfg.get('regression_flip_shift', False):
                    output_heatmap[..., 0] -= 1.0 / img_width
                output_heatmap = output_heatmap / 2

        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(
                img_metas, output_heatmap, img_size=[img_width, img_height])
            result.update(keypoint_result)

            if not return_heatmap:
                output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result

    def decode_images_to_heatmaps_minmax(self, images, resize=False):
        """

        Args:
            images: (bs, 256, 192, 3)
            resize: whether to resize to (64, 48)

        Returns:
            heatmaps: (bs, 17, h, w)
        """
        assert images.shape[-1] == 3
        batch_size, image_height, image_width, _ = images.shape
        images = images.float()

        # classify each pixel using GB
        GB = images[..., 1:].view(batch_size, 1, image_height, image_width, 2)  # (bs, 1, 256, 192, 2)
        colors = TopDown.colors
        num_classes = colors.shape[0]
        colors = colors.view(1, -1, 1, 1, 2)
        dist = torch.abs(GB - colors).sum(-1)  # (bs, 18, 256, 192)
        dist, indices = torch.min(dist, dim=1)  # (bs, 256, 192)
        keypoint_mask_list = []
        for idx in range(num_classes):
            mask = indices == idx  # (bs, 256, 192)
            keypoint_mask_list.append(mask)

        R = images[..., 0]  # (bs, 256, 192)
        heatmap_list = []
        for idx in range(num_classes):
            if idx == 17:
                continue
            mask = keypoint_mask_list[idx]
            heatmap = mask * R
            heatmap_list.append(heatmap.unsqueeze(1))
        heatmaps = torch.cat(heatmap_list, dim=1)

        if resize:
            raise NotImplementedError

        return heatmaps.cpu().numpy() / 255.
