#!/bin/bash

#SBATCH -J l2_displacement
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/naravich/nnunet_texture_ablations/slurm-logs/slurm-l2_displacement-%j.out

# Load environment
module load miniforge/24.3.0-0
source activate /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/.venv

# Default to patch1 if no argument provided
PATCH=${1:-patch1}

echo "=========================================================="
echo "Calculating L2 displacement for INTENSITY variants (${PATCH})..."
echo "=========================================================="
python /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/analysis/calculate_l2_displacement.py \
    --input intensity_variant_results_101-019_${PATCH}.pth

echo ""
echo "=========================================================="
echo "Calculating L2 displacement for TEXTURE variants (${PATCH})..."
echo "=========================================================="
python /nfs/erelab001/shared/Computational_Group/Naravich/nnUNet/analysis/calculate_l2_displacement.py \
    --input texture_variant_results_101-019_${PATCH}.pth
