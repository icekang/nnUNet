#!/bin/bash

# Activating the conda environment
source activate nnunet
which python

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"
# export nnUNet_def_n_proc=1

PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results/Dataset300_Lumen_and_Wall_OCT/nnUNetTrainer__nnUNetPreprocessPlans__2d/fold_all/checkpoint_best.pth
nnUNetv2_train 302 2d 1 -pretrained_weights $PATH_TO_CHECKPOINT