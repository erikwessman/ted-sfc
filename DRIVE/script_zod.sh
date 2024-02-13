#!/bin/bash

source activate pyRL

PHASE=$1
GPU_IDS=$2
NUM_WORKERS=$3
EXP_TAG=$4

LOG_DIR="./logs/${EXP_TAG}"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi

LOG="${LOG_DIR}/${PHASE}_${EXP_TAG}_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=$GPU_IDS

python main_sac.py --output ./output/${EXP_TAG} --phase ${PHASE} --num_workers ${NUM_WORKERS} --config cfgs/sac_ae_mlnet_zod.yml --gpu_id $GPU_IDS

echo "Done!"
