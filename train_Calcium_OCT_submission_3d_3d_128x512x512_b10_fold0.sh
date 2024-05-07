#!/bin/bash

# Activating the conda environment
source activate nnunet
which python

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"
# export nnUNet_def_n_proc=1

fold=0
CONFIG=3d_128x512x512_b10
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results/Dataset300_Lumen_and_Wall_OCT/nnUNetTrainer__nnUNetPreprocessPlans__3d_fullres/fold_all/checkpoint_best.pth
nnUNetv2_train 302 $CONFIG $fold -pretrained_weights $PATH_TO_CHECKPOINT

TRAINER=nnUNetTrainer
echo "FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
INPUT_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${fold}
LABEL_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs

nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $fold -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
echo "Completed FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
