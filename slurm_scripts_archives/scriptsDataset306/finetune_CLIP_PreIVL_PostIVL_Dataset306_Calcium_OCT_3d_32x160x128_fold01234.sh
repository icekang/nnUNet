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
DATASET_ID=306
TRAINER=nnUNetTrainer

PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/projects/OpenAI-CLIP/clip_pretrained_nnUNet.pt

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 0 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 1 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT ;

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 2 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 3 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT ;

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 4 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT ;

RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_Sohee_Ajay_Calcium_OCT/${TRAINER}__nnUNetPlans__${CONFIG}

mv ${RESULT_DIRECTORY_PATH} ${RESULT_DIRECTORY_PATH}_CLIP_PreIVL_PostIVL_pretrained