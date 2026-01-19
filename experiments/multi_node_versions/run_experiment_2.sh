#!/bin/bash
#SBATCH --job-name=exp2_mpi
#SBATCH --time=00:30:00
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=compute
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB

module load 2025
module load miniconda3
module load cuda/12.1
module load openmpi/4.1.7

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/mt

# Increased time limit and memory slightly compared to exp 1 just in case,
# as Exp 2 has multiple pipelines and might be heavier.

srun python experiments/multi_node_versions/experiment_2_mpi.py 
