#!/bin/bash

#SBATCH -J summarize_texture_ablations
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/naravich/nnunet_texture_ablations/slurm-logs/slurm-summarize-%j.out

# Load environment
module load miniforge/24.3.0-0
source activate /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/.venv

echo "Running summarized texture ablations analysis..."
python /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/analysis/summarize_texture_ablations.py
