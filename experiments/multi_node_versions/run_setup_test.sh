#!/bin/sh

#SBATCH --job-name=setup_test
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=00:10:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB



module load miniconda3
module load cuda/12.1


unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/mt

srun python experiments/multi_node_versions/verify_env.py > test_pyter.txt


