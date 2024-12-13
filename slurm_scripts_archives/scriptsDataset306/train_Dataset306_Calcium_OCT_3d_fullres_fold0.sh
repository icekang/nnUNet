#!/bin/bash

# Activating the conda environment
source activate nnunet
which python
export OPENBLAS_NUM_THREADS=1

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"
# export nnUNet_def_n_proc=1
export WANDB_DISABLED=True

fold=0
CONFIG=3d_fullres
DATASET_ID=306
TRAINER=nnUNetTrainerOriginal

nnUNetv2_train $DATASET_ID $CONFIG $fold -tr $TRAINER -device cuda # -pretrained_weights $PATH_TO_CHECKPOINT

CONFIG=2d
nnUNetv2_train $DATASET_ID $CONFIG $fold -tr $TRAINER -device cuda # -pretrained_weights $PATH_TO_CHECKPOINT
