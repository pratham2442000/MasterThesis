import pyterrier as pt
import torch
import sys
import logging

# Configure logging to write to a file instead of stdout
logging.basicConfig(
    filename='verify_env.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)

logging.info("--- Checking Environment ---")

# 1. Check GPU availability (PyTorch)
if torch.cuda.is_available():
    logging.info(f"[SUCCESS] CUDA is available.")
    logging.info(f"Device Name: {torch.cuda.get_device_name(0)}")
    logging.info(f"Device Count: {torch.cuda.device_count()}")
else:
    logging.info("[FAILURE] CUDA is NOT available. Check module load cuda or torch installation.")

# 2. Check PyTerrier Initialization (Requires Java)
try:
    if not pt.started():
        pt.init()
    logging.info("[SUCCESS] PyTerrier initialized successfully.")
except Exception as e:
    logging.info(f"[FAILURE] PyTerrier failed to initialize. Is Java loaded? Error: {e}")

# 3. Check simple retrieval to ensure index structures work
try:
    # A simple test dataframe
    import pandas as pd
    df = pd.DataFrame([["d1", "this is a test document"]], columns=["docno", "text"])
    indexer = pt.IterDictIndexer("./test_index", overwrite=True)
    index_ref = indexer.index(df.to_dict(orient="records"))
    logging.info("[SUCCESS] PyTerrier indexing worked.")
except Exception as e:
    logging.info(f"[FAILURE] PyTerrier indexing failed: {e}")