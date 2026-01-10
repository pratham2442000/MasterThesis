import pyterrier as pt
import torch
import sys

print("--- Checking Environment ---")

# 1. Check GPU availability (PyTorch)
if torch.cuda.is_available():
    print(f"[SUCCESS] CUDA is available.")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
else:
    print("[FAILURE] CUDA is NOT available. Check module load cuda or torch installation.")

# 2. Check PyTerrier Initialization (Requires Java)
try:
    if not pt.started():
        pt.init()
    print("[SUCCESS] PyTerrier initialized successfully.")
except Exception as e:
    print(f"[FAILURE] PyTerrier failed to initialize. Is Java loaded? Error: {e}")

# 3. Check simple retrieval to ensure index structures work
try:
    # A simple test dataframe
    import pandas as pd
    df = pd.DataFrame([["d1", "this is a test document"]], columns=["docno", "text"])
    indexer = pt.IterDictIndexer("./test_index", overwrite=True)
    index_ref = indexer.index(df.to_dict(orient="records"))
    print("[SUCCESS] PyTerrier indexing worked.")
except Exception as e:
    print(f"[FAILURE] PyTerrier indexing failed: {e}")