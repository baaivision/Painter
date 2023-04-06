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

from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures import Boxes, BoxMode, Instances, BitMasks, pairwise_iou

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    np.int
except:
    np.int = np.int32
    np.float = np.float32

from mmcv.ops import soft_nms

import sys
sys.path.append('.')
from util.matrix_nms import mask_matrix_nms
import data.register_coco_panoptic_annos_semseg


logger = logging.getLogger(__name__)


class COCOInstanceEvaluatorCustom(COCOEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(
            self,
            dataset_name: str,
            tasks=None,
            output_dir: Optional[str] = None,
            evaluator_inst = None,
            evaluator_semseg = None,
            label2cat = None,
            with_nms = False,
            nms_type = 'matrix',
            nms_iou = 0.6,
    ):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
            evaluator_inst
            evaluator_semseg
        """
        super().__init__(
            dataset_name=dataset_name,
            tasks=tasks,
            output_dir=output_dir,
        )
        self.evaluator_inst = evaluator_inst
        self.evaluator_semseg = evaluator_semseg
        self.file_path = None  # path to json format results for future Evaluation
        self.label2cat = label2cat
        self.with_nms = with_nms
        self.nms_type = nms_type
        self.nms_iou = nms_iou

    def process(self, inputs, outputs):

        for input, output in tqdm.tqdm(zip(inputs, outputs)):
            inst_seg_with_class = self.merge_inst_semseg_result(output)
            output = {"instances": inst_seg_with_class}

            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def merge_inst_semseg_result(self, output):
        inst_file = output['inst_file']
        semseg_file = output['semseg_file']
        # inst_image = Image.open(inst_file)
        semseg_image = Image.open(semseg_file)

        # obtaining semseg result is easy
        semseg_map, dist = self.evaluator_semseg.post_process_segm_output(
            np.array(semseg_image),  # (h, w), ndarray
        )

        # obtaining inst seg result is much more complex
        assert self.evaluator_inst.post_type == "threshold"
        output = self.evaluator_inst.post_process_segm_output_by_threshold(inst_file, keep_all=self.with_nms)

        inst_seg_with_class = self.merge_inst_semseg_result_to_instseg(semseg_map, dist, output['instances'])
        # inst_seg_with_class = output['instances']  # for check class-agnostic ap, checked

        # apply class-wise nms
        if self.with_nms:
            masks = inst_seg_with_class.pred_masks
            labels = inst_seg_with_class.pred_classes  # class-aware
            scores = inst_seg_with_class.scores

            if self.nms_type == 'matrix':
                scores, labels, masks, keep_inds = mask_matrix_nms(
                    masks=masks, labels=labels, scores=scores,
                    filter_thr=-1, nms_pre=-1, max_num=100,
                    kernel='gaussian', sigma=2.0, mask_area=None,
                )
            elif self.nms_type == 'soft':
                boxes = BitMasks(masks).get_bounding_boxes().tensor
                max_coordinate = boxes.max()
                offsets = labels.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
                boxes_for_nms = boxes + offsets[:, None]
                dets, keep = soft_nms(boxes=boxes_for_nms, scores=scores, iou_threshold=self.nms_iou,
                                      sigma=0.5, min_score=0.0, method="linear")
                boxes = boxes[keep]
                masks = masks[keep]
                labels = labels[keep]
                scores = dets[:, -1]  # scores are updated in soft-nms
            else:
                raise NotImplementedError(self.nms_type)

            # sort by score and keep topk
            num_pred = len(scores)
            topk = 100
            if num_pred > topk:
                _, topk_indices = scores.topk(topk, sorted=False)
                scores = scores[topk_indices]
                masks = masks[topk_indices]
                labels = labels[topk_indices]

            num_inst, height, width = masks.shape
            image_size = (height, width)
            result = Instances(image_size)
            result.pred_masks = masks.float()
            result.scores = scores
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
            result.pred_boxes = Boxes(torch.zeros(masks.shape[0], 4))
            result.pred_classes = labels
            inst_seg_with_class = result

        return inst_seg_with_class

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
        mask_probs = mask_probs.softmax(-1)
        # pred_classes = mask_probs.argmax(-1)
        probs, pred_classes = torch.max(mask_probs, dim=-1)

        # do not need to map id
        if self.label2cat is not None:
            pred_classes = torch.tensor(
                [self.label2cat[cls.item()] for cls in pred_classes],
                dtype=pred_classes.dtype, device=pred_masks.device)

        instance_seg.pred_classes = pred_classes
        return instance_seg


def get_args_parser():
    parser = argparse.ArgumentParser('COCO instance segmentation', add_help=False)
    parser.add_argument('--dist_thr', type=float, default=18.)
    parser.add_argument('--with_nms', action='store_true', default=False,
                        help="use keep_all inst, and merge semseg before applying nms")
    parser.add_argument('--nms_type', type=str, help='color type',
                        default='matrix', choices=['soft', 'matrix'])
    parser.add_argument('--nms_iou', type=float, default=0.6)
    parser.add_argument('--dist_type', type=str, help='color type',
                        default='abs', choices=['abs', 'square', 'mean'])
    parser.add_argument('--prompt', type=str, help='color type',
                        default="000000466730")
    parser.add_argument('--work_dir', type=str, help='color type',
                        default="models_inference/new3_all_lr5e-4/")
    parser.add_argument('--ckpt_file', type=str, default="")
    parser.add_argument('--input_size', type=int, default=448)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    # define pred paths
    ckpt_file = args.ckpt_file
    work_dir = args.work_dir
    pred_dir_inst = os.path.join(work_dir, 'pano_inst_inference_{}_{}_size{}'.format(
        ckpt_file, args.prompt, args.input_size))
    pred_dir_semseg = os.path.join(work_dir, "pano_semseg_inference_{}_{}_size{}".format(
        ckpt_file, args.prompt, args.input_size))
    gt_file = "datasets/coco/annotations/instances_val2017.json"

    print(pred_dir_inst)
    print(pred_dir_semseg)

    # define instance evaluator, note we only need the post-processing method
    dataset_name_inst = 'coco_2017_val'
    from eval.coco_panoptic.COCOCAInstSegEvaluatorCustom import COCOEvaluatorCustom
    # args_inst = get_inst_args()
    from eval.coco_panoptic.COCOCAInstSegEvaluatorCustom import define_colors_per_location_r_gb
    PALETTE_DICT_INST = define_colors_per_location_r_gb()
    evaluator_inst = COCOEvaluatorCustom(
        dataset_name_inst,
        tasks=("segm", ),
        # output_dir=None,
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
    dataset_name = 'coco_2017_val'
    output_dir = os.path.join(
        work_dir,
        "instance_segm_post_merge_{}_{}".format(ckpt_file, args.prompt),
        "dist{}_{}nms_iou{}".format(args.dist_thr, args.nms_type, args.nms_iou),
    )

    inputs = []
    outputs = []
    prediction_list_inst = glob.glob(os.path.join(pred_dir_inst, "*.png"))
    prediction_list_semseg = glob.glob(os.path.join(pred_dir_semseg, "*.png"))
    prediction_list_inst.sort()
    prediction_list_semseg.sort()
    print("num_pred: ", len(prediction_list_inst))
    print("loading predictions")
    coco_inst_annos = json.load(open(gt_file, 'r'))
    # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    label2cat = {label: cat_info['id'] for label, cat_info in enumerate(coco_inst_annos['categories'])}
    file_name_to_image_id = {image_info['file_name']: image_info['id'] for image_info in coco_inst_annos['images']}
    assert len(prediction_list_inst) == len(prediction_list_semseg) == len(file_name_to_image_id)
    for inst_file, semseg_file in zip(prediction_list_inst, prediction_list_semseg):
        assert os.path.basename(inst_file) == os.path.basename(semseg_file)
        file_name = os.path.basename(inst_file).replace('.png', '.jpg')
        image_id = file_name_to_image_id[file_name]
        # keys in input: "file_name", "image_id"
        input_dict = {"file_name": file_name, "image_id": image_id}
        # keys in output: "inst_file", "semseg_file"
        output_dict = {"inst_file": inst_file, "semseg_file": semseg_file}
        inputs.append(input_dict)
        outputs.append(output_dict)

    output_file = os.path.join(output_dir, "coco_instances_results.json")
    print("output file:", output_file)

    evaluator = COCOInstanceEvaluatorCustom(
        dataset_name=dataset_name, output_dir=output_dir, tasks=("segm", ),
        evaluator_inst=evaluator_inst, evaluator_semseg=evaluator_semseg,
        # label2cat=label2cat,
        label2cat=None,
        with_nms=args.with_nms,
        nms_type=args.nms_type,
        nms_iou=args.nms_iou,
    )
    evaluator.reset()
    evaluator.process(inputs, outputs)
    evaluator.evaluate()

    # get class-agnostic ap
    print("class-agnostic ap")
    cocoGt = COCO(annotation_file=gt_file)
    cocoDt = cocoGt.loadRes(output_file)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="segm")
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    results = cocoEval.stats

    # redo class-aware eval
    print("class-aware ap")
    cocoGt = COCO(annotation_file=gt_file)
    cocoDt = cocoGt.loadRes(output_file)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="segm")
    cocoEval.params.useCats = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    results = cocoEval.stats
    print(output_file)
