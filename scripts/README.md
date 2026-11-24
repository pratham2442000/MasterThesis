# Setup Scripts Guide

This directory contains scripts to prepare the environment, download datasets, and download models before running experiments.

## 1. `download_models.py`
Downloads required models from HuggingFace to the local cache directory for offline execution.
**Usage:**
```bash
python scripts/download_models.py
```
This downloads models specified in `constants.py` (Generator, Reranker, Dense Retrieval).

## 2. `dataset_setup.py`
Unified dataset setup script to build PyTerrier indexes and generate query rewrites.
**Usage:**
```bash
python scripts/dataset_setup.py                  # Run all steps
python scripts/dataset_setup.py --steps index    # Build BM25 index only
python scripts/dataset_setup.py --steps rewrites # Generate query rewrites only
python scripts/dataset_setup.py --steps dense    # Build dense index only
python scripts/dataset_setup.py --skip-dense     # Run all except dense
```

## 3. `download_data_ambiqa.py`
Downloads datasets and checkpoints from AmbigQA/NQ for DPR experiments.
**Usage:**
```bash
python scripts/download_data_ambiqa.py --resource <RESOURCE_NAME> --output_dir <DIR>
```
Run without args to see available resources.

## Pre-testing Checklist
1. Ensure `constants.py` has the correct paths, models, and dataset configuration.
2. Run `download_models.py` to get all necessary weights.
3. Run `dataset_setup.py` to prepare BM25, query rewrites, and dense indices.
4. (Optional) Run `download_data_ambiqa.py` if running AmbigQA/NQ experiments.
5. Execute experiments in `experiments/`.
