#!/bin/bash
#SBATCH --job-name="exp1_mpi"
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-faculty-dept
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load 2025
module load python
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate mt


module load openssh
module load git

# Source the environment setup script
# This handles module loading, conda env creation/activation, and dataset checking
source ./setup_env.sh


# Run the MPI script
# srun automatically handles the distribution of ranks based on --nodes and --ntasks-per-node
srun python experiments/experiment_1_mpi.py
