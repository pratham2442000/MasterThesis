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
if command -v module &> /dev/null; then
    module load miniconda3
    module load openssh
    module load git
else
    echo "Module command not found. Assuming local Conda or not on cluster."
fi

ENV_PATH="./scratch/${USER}/.conda/envs/mt"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating Conda environment in $ENV_PATH..."
    #running instal script
    bash setup_delftblue_env.sh
else
    echo "Conda environment exists at $ENV_PATH."
fi

# Activate environment
conda activate "$ENV_PATH"


# 3. Dataset Check
# Check default location first (local/dev)
# Then check /home location (cluster)

DEFAULT_DATA_PATH="/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA"
HOME_DATA_PATH="/home/${USER}/data/2wikimultihopQA"

if [ -d "$HOME_DATA_PATH" ]; then
    echo "Dataset found in $HOME_DATA_PATH. Setting DATA_PATH."
    export DATA_PATH="$HOME_DATA_PATH"
elif [ -d "$DEFAULT_DATA_PATH" ]; then
    echo "Dataset found in default path $DEFAULT_DATA_PATH."
    # No need to export if it matches the default in constants.py, but good for explicit behavior
    export DATA_PATH="$DEFAULT_DATA_PATH"
else
    echo "WARNING: Dataset not found in $HOME_DATA_PATH or $DEFAULT_DATA_PATH."
fi

echo "Environment setup complete. DATA_PATH is set to: ${DATA_PATH:-Default}"
