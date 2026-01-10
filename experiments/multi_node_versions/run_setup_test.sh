#!/bin/sh

#SBATCH --job-name=setup_test
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-msc-set
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB



module load java 
module load miniconda3
module load cuda/12.1


unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/mt

srun python verify_env.py > log.txt