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
module load openssh
module load git

bash setup_env.sh

# Run the MPI script
# srun automatically handles the distribution of ranks based on --nodes and --ntasks-per-node
srun python experiments/experiment_1_mpi.py
