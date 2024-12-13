#!/bin/bash

# Activating the conda environment
source activate nnunet
which python

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"
# export nnUNet_def_n_proc=1
export OPENBLAS_NUM_THREADS=1

CONFIG=3d_32x160x128_b10
DATASET_ID=302
TRAINER=nnUNetTrainerScaleAnalysis3

INPUT_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
LABEL_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs

PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/datasets/ModelGenesisOutputs/ModelGenesisNNUNetPretrainingV2_noNorm_correct_orientation/Converted_nnUNet_Genesis_OCT_Best.pt

for FOLD in 1 2
do
    TRAINING_OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
    OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_finetuned_Genesis_test

    echo "Begin training and evaluating FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER Genesis"
    nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -pretrained_weights $PATH_TO_CHECKPOINT

    nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth --save_probabilities
    nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
    echo "Renaming $TRAINING_OUTPUT_DIR to $TRAINING_OUTPUT_DIR"_Gensis""
    mv $TRAINING_OUTPUT_DIR $TRAINING_OUTPUT_DIR"_Gensis"
    echo "Completed FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER Genesis"
done
