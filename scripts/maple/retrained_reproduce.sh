#!/bin/bash

#cd ../..

# custom config
#DATA="/path/to/dataset/folder"
DATA="./data"
#TRAINER=MaPLe
TRAINER=CoCoOp

DATASET=$1
SEED=$2
WEIGHTSPATH=$3

#CFG=vit_b16_c2_ep5_batch4_2ctx
CFG=train_with_pretrained
SHOTS=16
#LOADEP=5
LOADEP=8
SUB_base=base
SUB_novel=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
#COMMON_DIR=
#MODEL_DIR=${WEIGHTSPATH}/base/seed${SEED}
MODEL_DIR=${WEIGHTSPATH}
DIR_base=output/base2new/finetune_test/test_${SUB_base}/${COMMON_DIR}
DIR_novel=output/base2new/finetune_test/test_${SUB_novel}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"
    # Evaluate on base classes
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_base} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_base}

    # Evaluate on novel classes
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR_novel} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB_novel}
fi
