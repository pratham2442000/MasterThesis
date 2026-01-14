#!/bin/bash
#SBATCH --job-name=exp1_mpi
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-dsait


module load miniconda3
module load cuda/12.1

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/mt

srun python srun python experiments/multi_node_versions/experiment_1_mpi.py > e1_pyter.txt
