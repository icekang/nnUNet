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

FOLD=1

TRAINER=nnUNetTrainer
CONFIG=3d_32x160x128_b10
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/datasets/ModelGenesisOutputs/ModelGenesisNNUNetPretrainingV2_noNorm_correct_orientation/Converted_nnUNet_Genesis_OCT_Best.pt
DATASET_ID=302

echo "Begin training and evaluating FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER from scratch"


nnUNetv2_train $DATASET_ID $CONFIG $FOLD -pretrained_weights $PATH_TO_CHECKPOINT

INPUT_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_finetuned_with_Genesis_test
LABEL_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs

nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1

echo "Renaming $TRAINING_OUTPUT_DIR to $TRAINING_OUTPUT_DIR"_finetuned_with_Genesis""
TRAINING_OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
mv $TRAINING_OUTPUT_DIR $TRAINING_OUTPUT_DIR"_finetuned_with_Genesis"

echo "Completed FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER"
