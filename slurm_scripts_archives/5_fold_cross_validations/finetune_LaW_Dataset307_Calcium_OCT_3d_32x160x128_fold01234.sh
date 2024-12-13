#!/bin/bash

# Activating the conda environment
source activate nnunet
which python
export OPENBLAS_NUM_THREADS=1

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

CONFIG=3d_32x160x128_b10
DATASET_ID=307
DATASET_NAME=Sohee_Calcium_OCT_CrossValidation
TRAINER=nnUNetTrainer

PRETRAIN_NAME=LaW_Pretrained
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results/Dataset300_Lumen_and_Wall_OCT/nnUNetTrainer__nnUNetPreprocessPlans__3d_fullres/fold_all/checkpoint_best.pth

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 0 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 1 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT ;

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 2 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 3 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT ;

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 4 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT ;

RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}

mv ${RESULT_DIRECTORY_PATH} ${RESULT_DIRECTORY_PATH}_${PRETRAIN_NAME}
