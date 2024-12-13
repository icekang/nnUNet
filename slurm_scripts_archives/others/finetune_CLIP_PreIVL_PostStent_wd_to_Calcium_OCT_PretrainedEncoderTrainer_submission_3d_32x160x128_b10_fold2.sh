#!/bin/bash

# Activating the conda environment
source activate nnunet
which python

# Setup env variables nn_UNet
# export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
# export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
# export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"
# export CUDA_VISIBLE_DEVICES=0
# export nnUNet_def_n_proc=1
export OPENBLAS_NUM_THREADS=1

PRETRAINING_STRATEGY=CLIP_PreIVL_PostStent_wd

FOLD=2
CONFIG=3d_32x160x128_b10
DATASET_ID=302
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/projects/OpenAI-CLIP/logs/${PRETRAINING_STRATEGY}/nnunet.pt
TRAINER=nnUNetTrainer4PretrainedEncoder
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -pretrained_weights $PATH_TO_CHECKPOINT -tr $TRAINER

echo "FOLD $FOLD CONFIG $CONFIG TRAINER $TRAINER"
INPUT_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_pretrained_with_${PRETRAINING_STRATEGY}_test
LABEL_DIR=$nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs

nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1

mv $nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD} $nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_pretrained_with_${PRETRAINING_STRATEGY}