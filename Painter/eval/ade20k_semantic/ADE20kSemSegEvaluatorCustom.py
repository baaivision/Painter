import glob
import json
import os
import argparse

import numpy as np
import torch
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from detectron2.evaluation import SemSegEvaluator


import sys
sys.path.insert(0, "./")

try:
    np.int
except:
    np.int = np.int32
    np.float = np.float32


def get_args_parser():
    parser = argparse.ArgumentParser('ADE20k semantic segmentation', add_help=False)
    parser.add_argument('--pred_dir', type=str, help='dir to ckpt', required=True)
    parser.add_argument('--dist_type', type=str, help='color type',
                        default='abs', choices=['abs', 'square', 'mean'])
    parser.add_argument('--suffix', type=str, help='model epochs',
                        default="default")
    return parser.parse_args()


class SemSegEvaluatorCustom(SemSegEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        palette=None,
        pred_dir=None,
        dist_type=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        super().__init__(
            dataset_name=dataset_name,
            distributed=distributed,
            output_dir=output_dir,
        )

        # update source names
        print(len(self.input_file_to_gt_file))
        self.input_file_to_gt_file_custom = {}
        for src_file, tgt_file in self.input_file_to_gt_file.items():
            assert os.path.basename(src_file).replace('.jpg', '.png') == os.path.basename(tgt_file)
            src_file_custom = os.path.join(pred_dir, os.path.basename(tgt_file))  # output is saved as png
            self.input_file_to_gt_file_custom[src_file_custom] = tgt_file

        color_to_idx = {}
        for cls_idx, color in enumerate(palette):
            color = tuple(color)
            # in ade20k, foreground index starts from 1
            color_to_idx[color] = cls_idx + 1
        self.color_to_idx = color_to_idx
        self.palette = torch.tensor(palette, dtype=torch.float, device="cuda")  # (num_cls, 3)
        self.pred_dir = pred_dir
        self.dist_type = dist_type

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        print("processing")
        for input in tqdm.tqdm(inputs):
            # output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)  # chw --> hw
            output = input["file_name"]
            output = Image.open(output)
            output = np.array(output)  # (h, w, 3)
            pred = self.post_process_segm_output(output)
            # use custom input_file_to_gt_file mapping
            gt_filename = self.input_file_to_gt_file_custom[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def post_process_segm_output(self, segm):
        """
        Post-processing to turn output segm image to class index map

        Args:
            segm: (H, W, 3)

        Returns:
            class_map: (H, W)
        """
        segm = torch.from_numpy(segm).float().to(self.palette.device)  # (h, w, 3)
        # pred = torch.einsum("hwc, kc -> hwk", segm, self.palette)  # (h, w, num_cls)
        h, w, k = segm.shape[0], segm.shape[1], self.palette.shape[0]
        if self.dist_type == 'abs':
            dist = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
        elif self.dist_type == 'square':
            dist = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
        elif self.dist_type == 'mean':
            dist_abs = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
            dist_square = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
            dist = (dist_abs + dist_square) / 2.
        else:
            raise NotImplementedError
        dist = torch.sum(dist, dim=-1)
        pred = dist.argmin(dim=-1).cpu()  # (h, w)
        pred = np.array(pred, dtype=np.int)

        return pred


if __name__ == '__main__':
    args = get_args_parser()
    dataset_name = 'ade20k_sem_seg_val'
    pred_dir = args.pred_dir
    suffix = args.suffix
    output_folder = os.path.join(pred_dir, 'eval_ade20k_{}'.format(suffix))

    from data.ade20k.gen_color_ade20k_sem import define_colors_per_location_mean_sep
    PALETTE = define_colors_per_location_mean_sep()

    evaluator = SemSegEvaluatorCustom(
        dataset_name,
        distributed=True,
        output_dir=output_folder,
        palette=PALETTE,
        pred_dir=pred_dir,
        dist_type=args.dist_type,
    )

    inputs = []
    outputs = []
    prediction_list = glob.glob(os.path.join(pred_dir, "*.png"))
    print(len(prediction_list))
    print("loading predictions")
    for file_name in prediction_list:
        # keys in input: "file_name", keys in output: "sem_seg"
        input_dict = {"file_name": file_name}
        output_dict = {"sem_seg": file_name}
        inputs.append(input_dict)
        outputs.append(output_dict)

    evaluator.reset()
    evaluator.process(inputs, outputs)
    results = evaluator.evaluate()
    print(results)

    copy_paste_results = {}
    for key in ['mIoU', 'fwIoU', 'mACC', 'pACC']:
        copy_paste_results[key] = results['sem_seg'][key]
    print(copy_paste_results)

    result_file = os.path.join(output_folder, "results.txt")
    print("writing to {}".format(result_file))
    with open(result_file, 'w') as f:
        print(results, file=f)
        print(copy_paste_results, file=f)
