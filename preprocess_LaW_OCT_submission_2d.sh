#!/bin/bash

# Activating the conda environment
source activate nnunet
which python

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

# run the script
# nnUNetv2_preprocess -d 300 -plans_name nnUNetPreprocessPlans -c 2d 3d_fullres -np 8 4 --verbose
nnUNetv2_train 300 2d all -p nnUNetPreprocessPlans
