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

FOLD=2
CONFIG=3d_32x160x128_b10
DATASET_ID=307
TRAINER=nnUNetTrainerScaleAnalysis80
DATASET_NAME=Sohee_Calcium_OCT_CrossValidation
PRETRAIN_NAME=Genesis_Pretrained

RESULT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}
PATH_TO_CHECKPOINT=${datasets_path}/../ModelGenesisOutputs/ModelGenesisNNUNetPretrainingV2_noNorm_correct_orientation/Converted_nnUNet_Genesis_OCT_Best.pt

echo "Starting training with CONFIG=$CONFIG, DATASET_ID=$DATASET_ID, TRAINER=$TRAINER"

nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER -device cuda -pretrained_weights $PATH_TO_CHECKPOINT

mv ${RESULT_DIRECTORY_PATH}/fold_2 ${RESULT_DIRECTORY_PATH}/fold_2_${PRETRAIN_NAME}

echo "Finished training fold 2 saving to ${RESULT_DIRECTORY_PATH}/fold_2_${PRETRAIN_NAME}"
