#!/bin/bash

# Activate conda environment
conda activate nnunet
which python
export OPENBLAS_NUM_THREADS=1

# Setup nnUNet environment variables
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

CONFIG=3d_32x160x128_b10
DATASET_ID=307
TRAINER=nnUNetTrainer
DATASET_NAME=Sohee_Calcium_OCT_CrossValidation
PRETRAIN_NAME=No_Pretrained

echo "Starting training with CONFIG=$CONFIG, DATASET_ID=$DATASET_ID, TRAINER=$TRAINER"

# # Run the first pair of folds
# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 0 -tr $TRAINER -device cuda &  # Task 1 (Fold 0) runs on GPU 0
# PID1=$!  # Capture PID of Task 1
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 1 -tr $TRAINER -device cuda &  # Task 2 (Fold 1) runs on GPU 1
# PID2=$!  # Capture PID of Task 2

# wait $PID1 $PID2  # Wait for both tasks to complete

# Run the second pair of folds
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 2 -tr $TRAINER -device cuda --c # Task 3 (Fold 2) runs on GPU 0
PID3=$!  # Capture PID of Task 3
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID $CONFIG 3 -tr $TRAINER -device cuda &  # Task 4 (Fold 3) runs on GPU 1
# PID4=$!  # Capture PID of Task 4

wait $PID3
# wait $PID3 $PID4  # Wait for both tasks to complete

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID $CONFIG 4 -tr $TRAINER -device cuda  # Task 5 (Fold 4) runs on GPU 1

# Move results
RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}
if [ -d "${RESULT_DIRECTORY_PATH}" ]; then
    mv ${RESULT_DIRECTORY_PATH} ${RESULT_DIRECTORY_PATH}_${PRETRAIN_NAME}
    echo "Results moved to ${RESULT_DIRECTORY_PATH}_${PRETRAIN_NAME}"
else
    echo "Error: Result directory ${RESULT_DIRECTORY_PATH} does not exist."
fi
