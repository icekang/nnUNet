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
PRETRAIN_NAME=CLIP_PrePostIVL_Pretrained

RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}
PATH_TO_CHECKPOINT=/home/gridsan/nchutisilp/projects/OpenAI-CLIP/clip_pretrained_nnUNet.pt

echo "Starting training with CONFIG=$CONFIG, DATASET_ID=$DATASET_ID, TRAINER=$TRAINER"

# Run the first pair of folds
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 0 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT &  # Task 1 (Fold 0) runs on GPU 0
PID1=$!  # Capture PID of Task 1
cpu_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 1 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT &  # Task 2 (Fold 1) runs on GPU 1
PID2=$!  # Capture PID of Task 2

wait $PID1 $PID2  # Wait for both tasks to complete
mv ${RESULT_DIRECTORY_PATH}/fold_0 ${RESULT_DIRECTORY_PATH}/fold_0_${PRETRAIN_NAME}
mv ${RESULT_DIRECTORY_PATH}/fold_1 ${RESULT_DIRECTORY_PATH}/fold_1_${PRETRAIN_NAME}

# Run the second pair of folds
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 2 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT & # Task 3 (Fold 2) runs on GPU 0
PID3=$!  # Capture PID of Task 3
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 3 -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT &  # Task 4 (Fold 3) runs on GPU 1
PID4=$!  # Capture PID of Task 4

wait $PID3 $PID4  # Wait for both tasks to complete
mv ${RESULT_DIRECTORY_PATH}/fold_2 ${RESULT_DIRECTORY_PATH}/fold_2_${PRETRAIN_NAME}
mv ${RESULT_DIRECTORY_PATH}/fold_3 ${RESULT_DIRECTORY_PATH}/fold_3_${PRETRAIN_NAME}

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 4 -tr $TRAINER -device cuda  -pretrained_weights $PATH_TO_CHECKPOINT # Task 5 (Fold 4) runs on GPU 1
mv ${RESULT_DIRECTORY_PATH}/fold_4 ${RESULT_DIRECTORY_PATH}/fold_4_${PRETRAIN_NAME}
