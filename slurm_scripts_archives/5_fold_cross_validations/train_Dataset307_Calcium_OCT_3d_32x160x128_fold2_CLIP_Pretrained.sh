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
DATASET_ID=307
TRAINER=nnUNetTrainer
DATASET_NAME=Sohee_Calcium_OCT_CrossValidation
PRETRAIN_NAME=CLIP_Pretrained

RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/projects/OpenAI-CLIP/clip_pretrained_nnUNet.pt

echo "Starting training with CONFIG=$CONFIG, DATASET_ID=$DATASET_ID, TRAINER=$TRAINER"

# Run the first pair of folds
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 2 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT  # Task 1 (Fold 0) runs on GPU 0
mv ${RESULT_DIRECTORY_PATH}/fold_2 ${RESULT_DIRECTORY_PATH}/fold_2_${PRETRAIN_NAME}