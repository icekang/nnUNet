#!/bin/bash
 
#SBATCH -J perturbation_eval
#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/naravich/nnunet_texture_ablations/slurm-logs/slurm-perturbation_eval-%A_%a.out

# Load environment
module load miniforge/24.3.0-0
source activate /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/.venv

# Determine fold (prioritizes CLI argument, then SLURM array index, defaulting to 0)
FOLD=${1:-${SLURM_ARRAY_TASK_ID:-0}}

echo "Running perturbation evaluation for fold: ${FOLD}"

# Run the python script
python /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/analysis/perturbation_evaluation.py --fold "${FOLD}"

