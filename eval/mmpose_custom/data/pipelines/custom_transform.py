import os
import random
import warnings

import cv2
import numpy as np
from PIL import Image


def define_colors_gb_mean_sep(num_locations=17):
    num_sep_per_channel = int(num_locations ** (1 / 2)) + 1  # 5
    separation_per_channel = 256 // num_sep_per_channel  # 51

    color_dict = {}
    # R = G = B = 0
    # B += separation_per_channel  # offset for the first loop
    for location in range(num_locations):
        num_seq_g = location // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_g <= num_sep_per_channel) and (num_seq_b <= num_sep_per_channel)

        G = 255 - num_seq_g * separation_per_channel
        B = 255 - num_seq_b * separation_per_channel
        assert (G < 256) and (B < 256)
        assert (G >= 0) and (B >= 0)
        assert (G, B) not in color_dict.values()

        color_dict[location] = (G, B)
        # print(location, (num_seq_g, num_seq_b), (G, B))

    # colors = [v for k, v in color_dict.items()]
    # min values in gb: [51, 51]
    return color_dict


color_dict = define_colors_gb_mean_sep()


def encode_target_to_image(target, target_weight, target_dir, metas):
    if len(target.shape) == 3:
        return encode_rgb_target_to_image(
            target_kernel=target, target_class=target,
            target_weight_kernel=target_weight, target_weight_class=target_weight,
            target_dir=target_dir, metas=metas,
        )

    assert len(target.shape) == 4
    return encode_rgb_target_to_image(
        target_kernel=target[1], target_class=target[0],
        target_weight_kernel=target_weight[1], target_weight_class=target_weight[0],
        target_dir=target_dir, metas=metas,
    )


def check_input(target_weight, target, metas):
    if not ((target_weight.reshape(17, 1, 1) * target) == target).all():
        print("useful target_weight!")
        target = target_weight.reshape(17, 1, 1) * target
    # make sure the invisible part is weighted zero, and thus not shown in target
    if not (target_weight[np.sum(metas['joints_3d_visible'], axis=1) == 0] == 0).all():
        print(metas['image_file'], "may have joints_3d_visible problems!")


def encode_rgb_target_to_image(target_kernel, target_class, target_weight_kernel, target_weight_class, target_dir, metas):
    """

    Args:
        target: ndarray (17, 256, 192)
        target_weight: ndarray (17, 1)
        metas: dict

    Returns:
        an RGB image, R encodes heatmap, GB encodes class

    """
    check_input(target_weight_kernel, target_kernel, metas)
    check_input(target_weight_class, target_class, metas)

    # 1. handle kernel in R channel
    # get max value for collision area
    sum_kernel = target_kernel.max(0)  # (256, 192)
    max_kernel_indices = target_kernel.argmax(0)  # (256, 192)
    R = sum_kernel[:, :, None] * 255.  # (256, 192, 1)

    # 2. handle class in BG channels
    K, H, W = target_class.shape
    keypoint_areas_class = []
    for keypoint_idx in range(K):
        mask = target_class[keypoint_idx] != 0
        keypoint_areas_class.append(mask)
    keypoint_areas_class = np.stack(keypoint_areas_class)  # (17, 256, 192)
    num_pos_per_location_class = keypoint_areas_class.sum(0)  # (256, 192)
    collision_area_class = num_pos_per_location_class > 1  # (256, 192)

    GB_MultiChannel = np.zeros((17, 256, 192, 2))
    for keypoint_idx in range(K):
        color = color_dict[keypoint_idx]
        class_mask = keypoint_areas_class[keypoint_idx]
        GB_MultiChannel[keypoint_idx][class_mask] = color
    GB = GB_MultiChannel.sum(0)  # (256, 192, 2)

    if np.sum(collision_area_class) != 0:
        for keypoint_idx in range(K):
            color = color_dict[keypoint_idx]
            # mach more max_area_this_keypoint for 0, but removed by collision_area_class latter
            max_area_this_keypoint = max_kernel_indices == keypoint_idx
            area_of_interest = max_area_this_keypoint * collision_area_class
            if not (area_of_interest == 0).all():
                GB[area_of_interest] = color

    # 3. get images / labels and save
    image_label = np.concatenate([R, GB], axis=-1).astype(np.uint8)  # (256, 192, 3)
    image_label = Image.fromarray(image_label)
    image = metas['img']
    image = Image.fromarray(image)

    box_idx = metas['bbox_id']

    _, filename = os.path.dirname(metas['image_file']), os.path.basename(metas['image_file'])
    image_path = os.path.join(target_dir, filename.replace(".jpg", "_box{}_image.png".format(box_idx)))
    label_path = os.path.join(target_dir, filename.replace(".jpg", "_box{}_label.png".format(box_idx)))

    # if os.path.exists(image_path):
    #     print(image_path, "exist! return!")
    #     return

    image.save(image_path)
    image_label.save(label_path)
