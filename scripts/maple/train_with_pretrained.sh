#!/bin/bash

#cd ../..

# Custom config
#DATA="/path/to/dataset/folder"
DATA="./data"
TRAINER=CoCoOp

DATASET=$1
SEED=$2
PRETRAINED_MODEL_PATH=$3

#CFG=vit_b16_c2_ep5_batch4_2ctx
CFG=train_with_pretrained
SHOTS=16

DIR=output/base2new/train_base/finetune/seed${SEED}/time6
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --resume ${PRETRAINED_MODEL_PATH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --resume ${PRETRAINED_MODEL_PATH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi

