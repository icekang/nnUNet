#!/bin/bash

# Job Flags
#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH --time=04:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=slurm-nnUNet-%x-%j.out

# Activating the conda environment
module load miniforge/24.3.0-0
source activate /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/.venv
which python

# Setup env variables nn_UNet
datasets_path="/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets"
export nnUNet_raw="$datasets_path/nnUNet_raw"
export nnUNet_preprocessed="$datasets_path/nnUNet_preprocessed"
export nnUNet_results="$datasets_path/nnUNet_results"

FOLD=1
CONFIG=3d_32x160x128_b10
DATASET_ID=310
TRAINER=nnUNetTrainerScaleAnalysis80
DATASET_NAME=nnInteractive_Calcium_OCT_CrossValidation
PRETRAIN_NAME=No_Pretrained

RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}

echo "Starting training with CONFIG=$CONFIG, DATASET_ID=$DATASET_ID, TRAINER=$TRAINER"

# Run the first pair of folds
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -device cuda
# PID1=$!  # Capture PID of Task 1
# wait $PID1

# wait $PID1 $PID2  # Wait for both tasks to complete
mv ${RESULT_DIRECTORY_PATH}/fold_${FOLD} ${RESULT_DIRECTORY_PATH}/fold_${FOLD}_${PRETRAIN_NAME}
echo "Finished training fold ${FOLD} saving to ${RESULT_DIRECTORY_PATH}/fold_${FOLD}_${PRETRAIN_NAME}"
