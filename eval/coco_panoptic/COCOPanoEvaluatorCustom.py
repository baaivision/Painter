# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import argparse
import glob
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional

import torch
import tqdm
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.structures import Instances, Boxes

from detectron2.evaluation import COCOPanopticEvaluator
# from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs

import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    np.int
except:
    np.int = np.int32
    np.float = np.float32

import sys
sys.path.append('.')

import data.register_coco_panoptic_annos_semseg

logger = logging.getLogger(__name__)


def combine_semantic_and_instance_outputs_custom(
    instance_results,
    semantic_results,
    overlap_threshold,
    stuff_area_thresh,
    instances_score_thresh,
):
    """
    Implement a simple combining logic following
    "combine_semantic_and_instance_predictions.py" in panopticapi
    to produce panoptic segmentation outputs.

    Args:
        instance_results: output of :func:`detector_postprocess`.
        semantic_results: an (H, W) tensor, each element is the contiguous semantic
            category id

    Returns:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
        segments_info (list[dict]): Describe each segment in `panoptic_seg`.
            Each dict contains keys "id", "category_id", "isthing".
    """
    panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)

    # sort instance outputs by scores
    sorted_inds = torch.argsort(-instance_results.scores)

    current_segment_id = 0
    segments_info = []

    instance_masks = instance_results.pred_masks.to(dtype=torch.bool, device=panoptic_seg.device)

    # Add instances one-by-one, check for overlaps with existing ones
    for inst_id in sorted_inds:
        score = instance_results.scores[inst_id].item()
        if score < instances_score_thresh:
            break
        mask = instance_masks[inst_id]  # H,W
        mask_area = mask.sum().item()

        if mask_area == 0:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum().item()

        if intersect_area * 1.0 / mask_area > overlap_threshold:
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": True,
                "score": score,
                "category_id": instance_results.pred_classes[inst_id].item(),
                "instance_id": inst_id.item(),
            }
        )

    # Add semantic results to remaining empty areas
    semantic_labels = torch.unique(semantic_results).cpu().tolist()
    for semantic_label in semantic_labels:
        # if semantic_label == 0:  # 0 is a special "thing" class
        #     continue
        if semantic_label < 80:  # all ids smaller than 80 are "thing" classes
            continue
        mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
        mask_area = mask.sum().item()
        if mask_area < stuff_area_thresh:
            continue

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": False,
                "category_id": semantic_label,
                "area": mask_area,
            }
        )

    return panoptic_seg, segments_info


class COCOPanopticEvaluatorCustom(COCOPanopticEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(
            self,
            dataset_name: str,
            output_dir: Optional[str] = None,
            evaluator_inst = None,
            evaluator_semseg = None,
            instance_seg_result_path = None,
            overlap_threshold = None,
            stuff_area_thresh = None,
            instances_score_thresh = None,
    ):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
            evaluator_inst
            evaluator_semseg
        """
        super().__init__(dataset_name=dataset_name, output_dir=output_dir)
        self.evaluator_inst = evaluator_inst
        self.evaluator_semseg = evaluator_semseg
        self.instance_seg_result_path = instance_seg_result_path
        self.cocoDt = None
        if self.instance_seg_result_path is not None:
            gt_file = "datasets/coco/annotations/instances_val2017.json"
            cocoGt = COCO(annotation_file=gt_file)
            inst_result_file = os.path.join(instance_seg_result_path, "coco_instances_results.json")
            print("loading pre-computed instance seg from \n{}".format(inst_result_file))
            cocoDt = cocoGt.loadRes(inst_result_file)
            self.cocoDt = cocoDt
            self.cat2label = {cat_info['id']: label for label, cat_info in enumerate(cocoGt.dataset['categories'])}
        self.overlap_threshold = overlap_threshold
        self.stuff_area_thresh = stuff_area_thresh
        self.instances_score_thresh = instances_score_thresh

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in tqdm.tqdm(zip(inputs, outputs)):
            # panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img, segments_info = self.merge_inst_semseg_result_to_panoseg(output)
            panoptic_img = panoptic_img.cpu().numpy()
            assert segments_info is not None

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def merge_inst_semseg_result_to_panoseg(self, output):
        # keys in segments_info:
        #   {
        #   "id": int(panoptic_label) + 1,
        #   "category_id": int(pred_class),
        #   "isthing": bool(isthing),
        #   }
        inst_file = output['inst_file']
        semseg_file = output['semseg_file']
        # inst_image = Image.open(inst_file)
        semseg_image = Image.open(semseg_file)
        # obtaining semseg result is easy
        semseg_map, dist = self.evaluator_semseg.post_process_segm_output(
            np.array(semseg_image),  # (h, w), ndarray
        )

        # obtaining inst seg result is much more complex
        if self.cocoDt is None:
            if self.evaluator_inst.post_type == "minmax":
                output_instance, dist_map, pred_map = self.evaluator_inst.post_process_segm_output_by_minmax(inst_file)
            elif self.evaluator_inst.post_type == "threshold":
                output_instance = self.evaluator_inst.post_process_segm_output_by_threshold(inst_file)
            else:
                raise NotImplementedError
            inst_seg_with_class = self.merge_inst_semseg_result_to_instseg(semseg_map, dist, output_instance['instances'])
        else:
            # load pre-computed dt
            image_id = output["image_id"]
            instance_det = self.cocoDt.imgToAnns[image_id]
            scores = [det['score'] for det in instance_det]
            segmentations = [det['segmentation'] for det in instance_det]
            category_ids = [self.cat2label[det['category_id']] for det in instance_det]

            scores = torch.tensor(scores, device="cuda")
            category_ids = torch.tensor(category_ids, device="cuda")
            segmentations = mask_utils.decode(segmentations)
            height, width, num_inst = segmentations.shape
            segmentations = torch.tensor(segmentations, device="cuda").permute(2, 0, 1).contiguous()

            result = Instances((height, width))
            result.pred_masks = segmentations.float()
            result.scores = scores
            result.pred_boxes = Boxes(torch.zeros(num_inst, 4))
            result.pred_classes = category_ids
            output_instance = {'instances': result}
            inst_seg_with_class = output_instance['instances']

        panoptic_img, segments_info = combine_semantic_and_instance_outputs_custom(
            instance_results=inst_seg_with_class,
            semantic_results=torch.from_numpy(semseg_map).to(inst_seg_with_class.pred_classes.device),
            overlap_threshold=self.overlap_threshold,
            stuff_area_thresh=self.stuff_area_thresh,
            instances_score_thresh=self.instances_score_thresh,
        )
        return panoptic_img, segments_info

    def merge_inst_semseg_result_to_instseg(self, semseg_map, semseg_dist, instance_seg):
        """
        label each instance via max vote
        Args:
            semseg_map: (h, w)
            semseg_dist: (h, w, num_cls)
            instance_seg: Instances with fields dict_keys(['pred_masks', 'scores', 'pred_boxes', 'pred_classes'])
        Returns:
            instance_seg_with_class
        """
        pred_masks = instance_seg.pred_masks  # (num_inst, h, w)
        semseg_dist = torch.from_numpy(semseg_dist).to(pred_masks.device)[:, :, :80]  # select from the best thing class
        semseg_prob = 1. - semseg_dist / torch.max(semseg_dist)  # (h, w, k)
        mask_probs = torch.einsum("nhw, hwk -> nk", pred_masks, semseg_prob)
        pred_classes = mask_probs.argmax(-1)

        instance_seg.pred_classes = pred_classes
        return instance_seg


def get_args_parser_pano_seg():
    parser = argparse.ArgumentParser('COCO panoptic segmentation', add_help=False)
    parser.add_argument('--dist_type', type=str, help='color type',
                        default='abs', choices=['abs', 'square', 'mean'])
    parser.add_argument('--prompt', type=str, help='color type',
                        default="000000466730")
    parser.add_argument('--ckpt_file', type=str, default="")
    parser.add_argument('--overlap_threshold', type=float, default=0.5)
    parser.add_argument('--stuff_area_thresh', type=float, default=8192)
    parser.add_argument('--instances_score_thresh', type=float, default=0.55)
    # args for inst results
    parser.add_argument('--dist_thr', type=float, default=16.)
    parser.add_argument('--nms_type', type=str, help='color type',
                        default='matrix', choices=['soft', 'matrix'])
    parser.add_argument('--nms_iou', type=float, default=0.6)
    parser.add_argument('--work_dir', type=str, help='color type',
                        default="")
    parser.add_argument('--input_size', type=int, default=448)
    return parser.parse_args()


if __name__ == "__main__":
    # pano args
    args = get_args_parser_pano_seg()
    print(args)
    ckpt_file = args.ckpt_file
    # define pred paths
    work_dir = args.work_dir
    pred_dir_inst = os.path.join(work_dir, "pano_inst_inference_{}_{}_size{}".format(
        ckpt_file, args.prompt, args.input_size))
    pred_dir_semseg = os.path.join(work_dir, "pano_semseg_inference_{}_{}_size{}".format(
        ckpt_file, args.prompt, args.input_size))
    instance_seg_result_path = os.path.join(
        work_dir,
        "instance_segm_post_merge_{}_{}".format(ckpt_file, args.prompt),
        "dist{}_{}nms_iou{}".format(args.dist_thr, args.nms_type, args.nms_iou),
    )
    gt_file = "datasets/coco/annotations/instances_val2017.json"

    print(pred_dir_inst)
    print(pred_dir_semseg)

    # define instance evaluator, note we only need the post-processing method
    dataset_name_inst = 'coco_2017_val'
    from eval.coco_panoptic.COCOCAInstSegEvaluatorCustom import COCOEvaluatorCustom
    from eval.coco_panoptic.COCOCAInstSegEvaluatorCustom import define_colors_per_location_r_gb
    PALETTE_DICT_INST = define_colors_per_location_r_gb()
    evaluator_inst = COCOEvaluatorCustom(
        dataset_name_inst,
        tasks=("segm", ),
        palette_dict=PALETTE_DICT_INST,
        pred_dir=pred_dir_inst,
        num_windows=4,
        post_type="threshold",
        dist_thr=args.dist_thr,
    )

    # define semantic seg evaluator, note we only need the post-processing method
    dataset_name_semseg = 'coco_2017_val_panoptic_with_sem_seg'
    from eval.coco_panoptic.COCOPanoSemSegEvaluatorCustom import SemSegEvaluatorCustom
    # args_semseg = get_semseg_args()
    from data.coco_semseg.gen_color_coco_panoptic_segm import define_colors_by_mean_sep
    PALETTE_DICT_SEMSEG = define_colors_by_mean_sep()
    PALETTE_SEMSEG = [v for k, v in PALETTE_DICT_SEMSEG.items()]
    evaluator_semseg = SemSegEvaluatorCustom(
        dataset_name_semseg,
        distributed=True,
        palette=PALETTE_SEMSEG,
        pred_dir=pred_dir_semseg,
        dist_type="abs",
    )

    # define pano seg evaluator
    # dataset_name = 'coco_2017_val_panoptic'
    dataset_name = 'coco_2017_val_panoptic_with_sem_seg'
    output_dir = os.path.join(work_dir, "panoptic_segm_{}_OverlapThr{}_StuffAreaThr{}_InstScoreThr{}".format(
        ckpt_file, args.overlap_threshold, args.stuff_area_thresh, args.instances_score_thresh))

    evaluator = COCOPanopticEvaluatorCustom(
        dataset_name=dataset_name, output_dir=output_dir,
        evaluator_inst=evaluator_inst, evaluator_semseg=evaluator_semseg,
        instance_seg_result_path=instance_seg_result_path,
        overlap_threshold=args.overlap_threshold,
        stuff_area_thresh=args.stuff_area_thresh,
        instances_score_thresh=args.instances_score_thresh,
    )

    inputs = []
    outputs = []
    prediction_list_inst = glob.glob(os.path.join(pred_dir_inst, "*.png"))
    prediction_list_semseg = glob.glob(os.path.join(pred_dir_semseg, "*.png"))
    prediction_list_inst.sort()
    prediction_list_semseg.sort()
    print("num_pred: ", len(prediction_list_inst))
    print("loading predictions")
    coco_pano_annos = json.load(open(gt_file, 'r'))
    # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    file_name_to_image_id = {image_info['file_name']: image_info['id'] for image_info in coco_pano_annos['images']}
    assert len(prediction_list_inst) == len(prediction_list_semseg) == len(file_name_to_image_id)
    for inst_file, semseg_file in zip(prediction_list_inst, prediction_list_semseg):
        assert os.path.basename(inst_file) == os.path.basename(semseg_file)
        file_name = os.path.basename(inst_file).replace('.png', '.jpg')
        image_id = file_name_to_image_id[file_name]
        # keys in input: "file_name", "image_id"
        input_dict = {"file_name": file_name, "image_id": image_id}
        # keys in output: "inst_file", "semseg_file"
        output_dict = {
            "file_name": file_name, "image_id": image_id,  # add the infos for loading pre-computed instances
            "inst_file": inst_file, "semseg_file": semseg_file,
        }
        inputs.append(input_dict)
        outputs.append(output_dict)

    evaluator.reset()
    evaluator.process(inputs, outputs)
    results = evaluator.evaluate()
    print("all results:")
    print(results)
    print("\nPanoptic:")
    res = results["panoptic_seg"]
    for key in ["PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"]:
        print(key, res[key])
