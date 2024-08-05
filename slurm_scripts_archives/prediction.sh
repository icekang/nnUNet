#!/bin/bash

source activate nnunet
which python

# Sohee_Stent_Toy/101-56.nii.gz STENT_0000_0000.nii.gz
# Sohee_Stent_Toy/101-26.nii.gz STENT_0001_0000.nii.gz
# Sohee_Stent_Toy/AU_MON_00005.nii.gz STENT_0002_0000.nii.gz
# Sohee_Stent_Toy/BE_OLV_00013.nii.gz STENT_0003_0000.nii.gz
# Sohee_Stent_Toy/BE_OLV_00031.nii.gz STENT_0004_0000.nii.gz

nnUNetv2_predict -i /home/gridsan/nchutisilp/projects/jepa/stent_segmentations_input_image -o /home/gridsan/nchutisilp/projects/jepa/stent_segmentations_LaW_prediction -d 300 -c 3d_fullres -f all -p nnUNetPreprocessPlans