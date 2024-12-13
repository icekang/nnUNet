#!/bin/bash

source activate nnunet
which python

export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1

nnUNetv2_predict -i /home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTr -o /home/gridsan/nchutisilp/datasets/SAM2_Dataset302_Calcium_OCTv2/Dataset302_Calcium_OCTv2_LaW_Prediction_imagesTr -d 300 -c 3d_fullres -f all -p nnUNetPreprocessPlans -device cuda -chk checkpoint_best.pth
