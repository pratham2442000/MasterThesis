#!/bin/bash
#SBATCH --job-name=exp2_mpi
#SBATCH --time=03:00:00
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4GB

module load 2025
module load miniconda3
module load openmpi/4.1.7

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/.conda/envs/mt

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun python experiments/multi_node_versions/experiment_2_mpi.py 

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"