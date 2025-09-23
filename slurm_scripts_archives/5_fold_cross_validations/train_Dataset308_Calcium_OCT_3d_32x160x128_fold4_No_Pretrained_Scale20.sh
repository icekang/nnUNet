#!/bin/bash

# Activate conda environment
source activate nnunet

which python
export OPENBLAS_NUM_THREADS=1

# Setup nnUNet environment variables
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

CONFIG=3d_32x160x128_b10
DATASET_ID=308
TRAINER=nnUNetTrainerScaleAnalysis20
DATASET_NAME=Sohee_Calcium_OCT_CrossValidation
PRETRAIN_NAME=No_Pretrained
FOLD=4
RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}

echo "Starting training with CONFIG=$CONFIG, DATASET_ID=$DATASET_ID, TRAINER=$TRAINER"

# Run the first pair of folds
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -device cuda

mv ${RESULT_DIRECTORY_PATH}/fold_${FOLD} ${RESULT_DIRECTORY_PATH}/fold_${FOLD}_${PRETRAIN_NAME}
