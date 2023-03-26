# !/bin/bash

set -x

NUM_GPUS=8
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=000000391460

SIZE=560
DIST_THR=19

CKPT_PATH="models/${JOB_NAME}/${CKPT_FILE}"
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

WORK_DIR="models_inference/${JOB_NAME}"

# inference
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_semseg.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=29504 --use_env \
 eval/coco_panoptic/painter_inference_pano_inst.py \
 --ckpt_path ${CKPT_PATH} --model ${MODEL} --prompt ${PROMPT} \
 --input_size ${SIZE}

# postprocessing and eval
python \
  eval/coco_panoptic/COCOInstSegEvaluatorCustom.py \
  --work_dir ${WORK_DIR} --ckpt_file ${CKPT_FILE} \
  --dist_thr ${DIST_THR} --prompt ${PROMPT} --input_size ${SIZE}

python \
  eval/coco_panoptic/COCOPanoEvaluatorCustom.py \
  --work_dir ${WORK_DIR} --ckpt_file ${CKPT_FILE} \
  --dist_thr ${DIST_THR} --prompt ${PROMPT} --input_size ${SIZE}
