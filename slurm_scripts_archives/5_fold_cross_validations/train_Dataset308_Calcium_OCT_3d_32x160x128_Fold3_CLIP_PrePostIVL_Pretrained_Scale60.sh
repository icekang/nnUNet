#!/bin/bash

# Activate conda environment
source activate nnunet

which python
export OPENBLAS_NUM_THREADS=1

# Setup nnUNet environment variables
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

CONFIG=3d_32x160x128_b10
DATASET_ID=308
TRAINER=nnUNetTrainerScaleAnalysis60
DATASET_NAME=Sohee_Calcium_OCT_CrossValidation
PRETRAIN_NAME=CLIP_PrePostIVL_Pretrained
FOLD=3

RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/projects/OpenAI-CLIP/clip_pretrained_nnUNet.pt

echo "Starting training with CONFIG=$CONFIG, DATASET_ID=$DATASET_ID, TRAINER=$TRAINER FOLD=$FOLD"

nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT

mv ${RESULT_DIRECTORY_PATH}/fold_${FOLD} ${RESULT_DIRECTORY_PATH}/fold_${FOLD}_${PRETRAIN_NAME}