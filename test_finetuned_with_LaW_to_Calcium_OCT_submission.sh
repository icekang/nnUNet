#!/bin/bash

# Activating the conda environment
source activate nnunet
which python

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

DATASET_ID=302 # Calcium
for fold in $(seq 1 2); do
    for CONFIG in 3d_fullres; do
        for TRAINER in nnUNetTrainer; do
            echo "FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
            INPUT_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
            OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${fold}_finetuned_with_LaW_test
            LABEL_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs

            nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $fold -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
            nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
            echo "Completed FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
        done
    done
done
