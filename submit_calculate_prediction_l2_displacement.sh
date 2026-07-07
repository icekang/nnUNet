#!/bin/bash

#SBATCH -J prediction_l2_displacement
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/naravich/nnunet_texture_ablations/slurm-logs/slurm-prediction_l2-%j.out

# Load environment
module load miniforge/24.3.0-0
source activate /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/.venv

echo "Calculating relative and absolute L2 displacement of predictions across all folds..."
python /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/analysis/calculate_prediction_l2_displacement.py
