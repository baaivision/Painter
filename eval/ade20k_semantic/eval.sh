# !/bin/bash

set -x

NUM_GPUS=4
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"
PROMPT=ADE_train_00009574

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="models/${JOB_NAME}/${CKPT_FILE}"
DST_DIR="models_inference/${JOB_NAME}/ade20k_semseg_inference_${CKPT_FILE}_${PROMPT}_size${SIZE}"

# inference
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=29504 --use_env \
  eval/ade20k_semantic/painter_inference_segm.py \
  --model ${MODEL} --prompt ${PROMPT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE}

# postprocessing and eval
python eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}
