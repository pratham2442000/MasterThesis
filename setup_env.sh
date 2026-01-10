#!/bin/bash

# 1. Repo Check & Clone
if [ ! -d ".git" ]; then
    echo "Git repository not found. Cloning..."
    # Check if directory is empty, if not, warn or try to clone into current dir if allowed
    if [ "$(ls -A)" ]; then
         echo "Current directory is not empty. Cloning might fail or be messy."
    fi
    git clone https://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@github.com/pratham2442000/MasterThesis . || echo "Clone failed or directory not empty. Continuing..."
else
    echo "Git repository found."
fi

# 2. Conda Setup
# Load miniconda module

module load miniconda3


ENV_PATH="/scratch/pjohari/.conda/envs/mt"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating Conda environment in $ENV_PATH..."
    #running instal script
    bash setup_delftblue_env.sh
else
    echo "Conda environment exists at $ENV_PATH."
fi

# Activate environment
conda activate $ENV_PATH

# 3. Dataset Check
# Check default location first (local/dev)
# Then check /home location (cluster)

HOME_DATA_PATH="/home/pjohari/data/2wikimultihopQA"

if [ -d "$HOME_DATA_PATH" ]; then
    echo "Dataset found in $HOME_DATA_PATH. Setting DATA_PATH."
    export DATA_PATH="$HOME_DATA_PATH"
else
    echo "WARNING: Dataset not found in $HOME_DATA_PATH."
fi

echo "Environment setup complete. DATA_PATH is set to: ${DATA_PATH:-Default}"
