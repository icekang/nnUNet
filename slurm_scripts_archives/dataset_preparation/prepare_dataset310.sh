#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=20G
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

nnUNetv2_plan_and_preprocess -d 310  --verify_dataset_integrity