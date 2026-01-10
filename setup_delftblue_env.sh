#!/bin/bash

# Setup script for PyTerrier environment on DelftBlue
# Usage: source setup_delftblue_env.sh

# 1. Load Modules
echo "Loading DelftBlue modules..."
module load 2024r1
module load miniconda3
module load cuda/12.1 # Using a recent CUDA version compatible with PyTorch
module load openjdk # Required for PyTerrier

# 2. Define Environment Path in Scratch
ENV_PATH="/scratch/${USER}/.conda/envs/mt"
echo "Environment will be created at: $ENV_PATH"

# 3. Create Conda Environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating Conda environment..."
    # Create environment with Python 3.10
    conda create -y -p "$ENV_PATH" python=3.10
    pip install -r /home/req.txt
else
    echo "Environment already exists."
fi

# 4. Activate Environment
echo "Activating environment..."
source activate "$ENV_PATH"


