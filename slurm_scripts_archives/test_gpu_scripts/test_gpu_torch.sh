#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 1
#SBATCH --mem=4G
#####SA SBATCH --gres=gpu:l40s:1
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

ls -lh $nnUNet_raw
ls -lh $nnUNet_preprocessed
ls -lh $nnUNet_results