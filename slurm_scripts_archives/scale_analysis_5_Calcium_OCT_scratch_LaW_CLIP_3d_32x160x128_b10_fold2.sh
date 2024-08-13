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

FOLD=2

CONFIG=3d_32x160x128_b10
DATASET_ID=302
TRAINER=nnUNetTrainerScaleAnalysis5

INPUT_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
LABEL_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs


TRAINING_OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_finetuned_from_scratch_test

echo "Begin training and evaluating FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER from scratch"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER
nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth --save_probabilities
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
echo "Renaming $OUTPUT_DIR to $OUTPUT_DIR"_from_scratch""
mv $TRAINING_OUTPUT_DIR $TRAINING_OUTPUT_DIR"_from_scratch"
echo "Completed FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER from scratch"


OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_finetuned_with_LaW_test
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results/Dataset300_Lumen_and_Wall_OCT/nnUNetTrainer__nnUNetPreprocessPlans__3d_fullres/fold_all/checkpoint_best.pth

echo "Begin training and evaluating FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER with pretrained LaW"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -pretrained_weights $PATH_TO_CHECKPOINT
nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth --save_probabilities
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
echo "Renaming $OUTPUT_DIR to $OUTPUT_DIR"_pretrained_LaW""
mv $TRAINING_OUTPUT_DIR $TRAINING_OUTPUT_DIR"_pretrained_LaW"
echo "Completed FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER with pretrained LaW"

OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_finetuned_with_CLIP_test
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/projects/OpenAI-CLIP/clip_pretrained_nnUNet.pt

echo "Begin training and evaluating FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER with pretrained CLIP"
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -pretrained_weights $PATH_TO_CHECKPOINT
nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth --save_probabilities
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
echo "Renaming $OUTPUT_DIR to $OUTPUT_DIR"_pretrained_CLIP""
mv $TRAINING_OUTPUT_DIR $TRAINING_OUTPUT_DIR"_pretrained_CLIP"
echo "Completed FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER with pretrained CLIP"
