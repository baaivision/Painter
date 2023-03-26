import glob
import json
import os
import argparse

import numpy as np
import pycocotools.mask as mask_util
import itertools
from detectron2.utils.file_io import PathManager
from detectron2.structures import Boxes, BoxMode, Instances, BitMasks, pairwise_iou

import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


import sys
sys.path.insert(0, "./")

from util.matrix_nms import mask_matrix_nms


def get_args_parser():
    parser = argparse.ArgumentParser('COCO class-agnostic instance segmentation', add_help=False)
    parser.add_argument('--pred_dir', type=str, help='dir to ckpt',
                        default=None)
    parser.add_argument('--post_type', type=str, help='type of post-processing',
                        default="threshold", choices=["minmax", "threshold"])
    parser.add_argument('--dist_thr', type=float, help='dir to ckpt',
                        default=19.)
    parser.add_argument('--num_windows', type=int, default=4)
    return parser.parse_args()


def define_colors_per_location_r_gb(num_location_r=16, num_location_gb=20):
    sep_r = 255 // num_location_r  # 255 for bigger sep to bg
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
                    # print(location, R, G, B)
    return color_dict


def load_image_with_retry(image_path):
    while True:
        try:
            img = Image.open(image_path)
            return img
        except OSError as e:
            print(f"Catched exception: {str(e)}. Re-trying...")
            import time
            time.sleep(1)


class COCOEvaluatorCustom(COCOEvaluator):
    def __init__(
        self,
        dataset_name,
        tasks=None,
        output_dir=None,
        palette_dict=None,
        pred_dir=None,
        num_windows=4,
        topk=100,
        post_type="minmax",
        dist_thr=5.,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            palette_dict: location to color
        """
        super().__init__(
            dataset_name=dataset_name,
            tasks=tasks,
            output_dir=output_dir,
        )

        self.post_type = post_type
        if not isinstance(dist_thr, list):
            dist_thr = [dist_thr]
        self.dist_thr_list = dist_thr
        self.location2color = palette_dict
        self.color2location = {v: k for k, v in self.location2color.items()}
        palette = [v for k, v in palette_dict.items()]
        palette.append((0, 0, 0))
        self.palette = torch.tensor(palette, dtype=torch.float, device="cuda")  # (num_cls, 3)
        self.pred_dir = pred_dir
        self.topk = topk
        self._do_evaluation = False  # we only save the results
        self.file_path = None  # path to json format results for future Evaluation
        self.num_windows = num_windows

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        # keys in input: "image_id",
        # keys in output: "instances", which contains pred_boxes, scores, pred_classes, pred_masks
        for input, output_raw in tqdm.tqdm(zip(inputs, outputs)):
            if self.post_type == "minmax":
                output, dist_map, pred_map = self.post_process_segm_output_by_minmax(output_raw['pred_path'])
            elif self.post_type == "threshold":
                output = self.post_process_segm_output_by_threshold(output_raw['pred_path'],
                                                                    dist_thr_list=self.dist_thr_list)
            else:
                raise NotImplementedError

            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        assert self._output_dir
        file_path = os.path.join(self._output_dir, "coco_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        assert self.file_path is None
        self.file_path = file_path
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        assert not self._do_evaluation
        return

    def post_process_segm_output_by_minmax(self, pred_path):
        """
        Post-processing to turn output segm image to class index map

        Args:
            pred_path:  path to a (H, W, 3) image

        Returns:
            class_map: (H, W)
        """
        # load prediction
        segm = load_image_with_retry(pred_path)
        height, width = segm.height, segm.width
        segm = np.array(segm)  # (h, w, 3)

        # get location cat for each pixel
        segm = torch.from_numpy(segm).float().to(self.palette.device)  # (h, w, 3)
        h, w, k = segm.shape[0], segm.shape[1], self.palette.shape[0]

        # dist = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k, 3)
        # dist = torch.sum(dist, dim=-1)  # (h, w, k)
        # # both (h, w), thus the k dim in dist is not needed, significantly reduce memory
        # dist_map, pred = torch.min(dist, dim=-1)

        # split window then merge
        dist_map_list = []
        pred_map_list = []
        # split in height
        window_size_h = h // self.num_windows + 1  # +1 to make sure no left
        for i in range(self.num_windows):
            h_start = i * window_size_h
            h_end = (i + 1) * window_size_h
            dist = torch.abs(segm.view(h, w, 1, 3)[h_start: h_end] - self.palette.view(1, 1, k, 3))  # (h, w, k, 3)
            dist = torch.sum(dist, dim=-1)  # (h, w, k)
            # both (h, w), thus the k dim in dist is not needed, significantly reduce memory
            dist_map, pred_map = torch.min(dist, dim=-1)
            dist_map_list.append(dist_map)
            pred_map_list.append(pred_map)
            # del dist
        dist_map = torch.cat(dist_map_list, dim=0)
        pred_map = torch.cat(pred_map_list, dim=0)
        assert dist_map.shape[0] == pred_map.shape[0] == h

        # get instances from the location cat map
        mask_list = []
        score_list = []
        class_list = []
        for location_cat in torch.unique(pred_map):
            if location_cat == len(self.palette) - 1:
                class_list.append(0)  # bg class will be ignored in eval
            else:
                class_list.append(1)
            mask = pred_map == location_cat  # (h, w)
            score_neg = torch.mean(dist_map[mask])
            mask_list.append(mask)
            score_list.append(score_neg)
        scores_neg = torch.stack(score_list)
        scores = 1 - scores_neg / max(torch.max(scores_neg), 1.)
        masks = torch.stack(mask_list)
        classes = torch.tensor(class_list, device=masks.device)

        # # sort by score and keep topk
        # num_pred = len(score_list)
        # if num_pred > self.topk:
        #     _, topk_indices = scores.topk(self.topk, sorted=False)
        #     scores = scores[topk_indices]
        #     masks = masks[topk_indices]

        image_size = (height, width)
        result = Instances(image_size)
        result.pred_masks = masks.float()
        result.scores = scores
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        result.pred_boxes = Boxes(torch.zeros(masks.shape[0], 4))
        result.pred_classes = classes

        output = {'instances': result}
        return output, dist_map, pred_map

    def post_process_segm_output_by_threshold(self, pred_path, dist_thr_list=None, keep_all=False):
        """
        Post-processing to turn output segm image to class index map

        Args:
            pred_path:  path to a (H, W, 3) image
            dist_thr_list
            keep_all: return all preds w/o nms and w/o top100

        Returns:
            class_map: (H, W)
        """
        if dist_thr_list is None:
            dist_thr_list = self.dist_thr_list
        # load prediction
        segm = load_image_with_retry(pred_path)
        height, width = segm.height, segm.width
        segm = np.array(segm)  # (h, w, 3)

        # get location cat for each pixel
        segm = torch.from_numpy(segm).float().to(self.palette.device)  # (h, w, 3)
        h, w, k = segm.shape[0], segm.shape[1], self.palette.shape[0]

        # make pred for each location category then merge
        mask_list = []
        dist_list = []
        maskness_neg_list = []

        all_palette = self.palette[:-1]
        num_color_each_time = 800  # +1 for bg
        num_parallels = int(all_palette.shape[0] // num_color_each_time) + 1
        for dist_thr in dist_thr_list:
            for idx in range(num_parallels):
                start_idx = idx * num_color_each_time
                end_idx = (idx + 1) * num_color_each_time
                color = all_palette[start_idx:end_idx]  # (num_color, 3)
                dist = torch.abs(segm.view(1, h, w, 3) - color.view(-1, 1, 1, 3))  # (num_color, h, w, 3)
                dist = torch.sum(dist, dim=-1) / 3.  # (num_color, h, w)
                mask = dist < dist_thr  # (num_color, h, w)
                num_pos = mask.sum((1, 2))
                keep = num_pos > 0
                mask = mask[keep]
                dist = dist[keep]
                if len(dist) > 0:
                    maskness_neg = (dist * mask.float()).sum((1, 2)) / (mask.sum((1, 2)))  # (num_color[keep], )
                    mask_list.append(mask)
                    # dist_list.append(dist)  # keep dist for debug only
                    maskness_neg_list.append(maskness_neg)

        # handle cases of empty pred
        if len(mask_list) == 0:
            image_size = (height, width)
            result = Instances(image_size)
            result.pred_masks = torch.zeros(1, height, width)
            result.scores = torch.zeros(1)
            result.pred_boxes = Boxes(torch.zeros(1, 4))
            result.pred_classes = torch.zeros(1)
            output = {'instances': result}
            return output

        # dists = torch.cat(dist_list, dim=0)  # (num_inst, h, w)
        masks = torch.cat(mask_list, dim=0)  # (num_inst, h, w)

        maskness_neg = torch.cat(maskness_neg_list, dim=0)  # (num_inst, )

        # the first sort before nms for keeping topk
        topk = 2000
        maskness_neg, indices = torch.sort(maskness_neg, descending=False)
        masks = masks[indices]
        masks = masks[:topk]
        maskness_neg = maskness_neg[:topk]  # (topk, h, w)

        # get scores
        scores = 1 - maskness_neg / max(torch.max(maskness_neg), 1.)  # (topk,)
        labels = torch.ones(masks.shape[0], device=masks.device)

        if not keep_all:
            # apply mask nms here
            scores, labels, masks, keep_inds = mask_matrix_nms(
                masks=masks, labels=labels, scores=scores,
                filter_thr=-1, nms_pre=-1, max_num=100,
                kernel='gaussian', sigma=2.0, mask_area=None,
            )

            # sort by score and keep topk
            num_pred = len(scores)
            if num_pred > self.topk:
                _, topk_indices = scores.topk(self.topk, sorted=False)
                scores = scores[topk_indices]
                masks = masks[topk_indices]
                labels = labels[topk_indices]

        image_size = (height, width)
        result = Instances(image_size)
        result.pred_masks = masks.float()
        result.scores = scores
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        result.pred_boxes = Boxes(torch.zeros(masks.shape[0], 4))
        result.pred_classes = labels

        output = {'instances': result}
        return output


if __name__ == '__main__':
    args = get_args_parser()
    dataset_name = 'coco_2017_val'
    coco_annotation = "datasets/coco/annotations/instances_val2017.json"

    pred_dir = args.pred_dir
    output_folder = os.path.join(pred_dir, 'eval_{}'.format(dataset_name))

    # get palette
    PALETTE_DICT = define_colors_per_location_r_gb()

    evaluator = COCOEvaluatorCustom(
        dataset_name,
        tasks=("segm", ),
        output_dir=output_folder,
        palette_dict=PALETTE_DICT,
        pred_dir=pred_dir,
        num_windows=args.num_windows,
        post_type=args.post_type,
        dist_thr=args.dist_thr,
    )

    cocoGt = COCO(annotation_file=coco_annotation)
    id2img = cocoGt.imgs
    img2id = {v['file_name']: k for k, v in id2img.items()}

    inputs = []
    outputs = []
    prediction_list = glob.glob(os.path.join(pred_dir, "*.png"))
    print("num_pred: ", len(prediction_list))
    print("loading predictions")
    for file_name in prediction_list:
        # keys in input: "image_id",
        # keys in output: "instances", which contains pred_boxes, scores, pred_classes, pred_masks
        image_org_name = os.path.basename(file_name).split("_")[0]
        image_org_name = image_org_name.replace(".png", ".jpg") if image_org_name.endswith(".png") \
            else image_org_name + ".jpg"  # else for gt eval
        image_id = img2id[image_org_name]
        input_dict = {"image_id": image_id}
        output_dict = {"pred_path": file_name}
        inputs.append(input_dict)
        outputs.append(output_dict)

    evaluator.reset()
    evaluator.process(inputs, outputs)
    evaluator.evaluate()

    # load result file and eval using cocoapi
    resFile = evaluator.file_path
    cocoDt = cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="segm")
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    results = cocoEval.stats
