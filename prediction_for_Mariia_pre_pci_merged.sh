#!/bin/bash

source activate nnunet
which python

export CUDA_VISIBLE_DEVICES=0
nnUNetv2_predict -i /home/gridsan/nchutisilp/datasets/Mariia/P3_MIT/pre_pci -o /home/gridsan/nchutisilp/datasets/Mariia/P3_MIT/pre_pci_prediction_LaW_merged304_all -d 304 -c 3d_fullres -f all -p nnUNetPlans

nnUNetv2_predict -i /home/gridsan/nchutisilp/datasets/Mariia/P3_MIT/pre_pci -o /home/gridsan/nchutisilp/datasets/Mariia/P3_MIT/pre_pci_prediction_LaW_all -d 300 -c 3d_fullres -f all -p nnUNetPreprocessPlans