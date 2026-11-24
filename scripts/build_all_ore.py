import os
import sys
import torch

from tqdm import tqdm

# ============================================================================
# OFFLINE MODE (IMPORTANT FOR CLUSTERS / SLURM / COMPUTE NODES)
# ============================================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

os.environ["TOKENIZERS_PARALLELISM"] = "true"

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.set_num_threads(os.cpu_count())

# ============================================================================
# PATH SETUP
# ============================================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ============================================================================
# IMPORTS
# ============================================================================

import pyterrier as pt

from pyterrier_dr import (
    FlexIndex,
    TasB,
    TctColBert,
)

from pyterrier_adaptive import (
    CorpusGraph,
    Laff,
)

import constants

from code_base.dataset_factory import (
    get_corpus_iter,
    get_dataset,
)

# ============================================================================
# PYTERRIER INIT
# ============================================================================

if not pt.started():

    # Prevent Maven resolution on offline nodes
    pt.terrier.set_version("5.11")
    pt.terrier.set_helper_version("0.0.8")

    pt.java.set_memory_limit(32000)

    pt.init()

# ============================================================================
# HARDWARE SETUP
# ============================================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if torch.cuda.is_available():

    gpu_name = torch.cuda.get_device_name(0)

    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
    )

    # Dense retrieval encoding batch sizes
    if total_vram_gb >= 22:
        DENSE_BATCH_SIZE = 512
        LAFF_BATCH_SIZE = 4096

    elif total_vram_gb >= 14:
        DENSE_BATCH_SIZE = 256
        LAFF_BATCH_SIZE = 2048

    elif total_vram_gb >= 10:
        DENSE_BATCH_SIZE = 128
        LAFF_BATCH_SIZE = 1024

    else:
        DENSE_BATCH_SIZE = 64
        LAFF_BATCH_SIZE = 512

else:

    gpu_name = "CPU"

    DENSE_BATCH_SIZE = 4
    LAFF_BATCH_SIZE = 4

print("=" * 80)
print(f"Running on: {device}")
print(f"GPU: {gpu_name}")
print(f"Dense Batch Size: {DENSE_BATCH_SIZE}")
print(f"LAFF Batch Size: {LAFF_BATCH_SIZE}")
print("=" * 80)

# ============================================================================
# DATASET
# ============================================================================

DATASET_NAME = constants.DATASET_NAME
SPLIT = constants.DATA_SPLIT

print("\nDataset Configuration")
print("-" * 80)
print(f"Dataset: {DATASET_NAME}")
print(f"Split: {SPLIT}")

# IMPORTANT:
# recreate iterator every use
def corpus():
    return get_corpus_iter(DATASET_NAME, SPLIT)

# ============================================================================
# CREATE BASE DIRECTORIES
# ============================================================================

os.makedirs("indices", exist_ok=True)

# ============================================================================
# LOAD DATASET (OPTIONAL)
# ============================================================================

print("\nLoading dataset metadata...")

docs_df, _, _ = get_dataset(DATASET_NAME, SPLIT)

if docs_df is not None:

    print(f"Loaded dataframe: {len(docs_df):,} documents")

else:

    print("Dataset loaded lazily via generator.")

# ============================================================================
# STEP 1 — BM25 INDEX
# ============================================================================

print("\n" + "=" * 80)
print("1. BM25 INDEXING")
print("=" * 80)

BM25_INDEX_DIR = constants.INDEX_PATH.replace(
    "/data.properties",
    ""
)

BM25_PROPERTIES = os.path.join(
    BM25_INDEX_DIR,
    "data.properties"
)

if os.path.exists(BM25_PROPERTIES):

    print("BM25 index already exists.")
    print(BM25_INDEX_DIR)

else:

    os.makedirs(BM25_INDEX_DIR, exist_ok=True)

    indexer = pt.IterDictIndexer(
        BM25_INDEX_DIR,
        overwrite=True,
        threads=os.cpu_count(),
        meta={
            "docno": 256,
            "text": 4096,
        },
    )

    print("\nStarting BM25 indexing...")

    index_ref = indexer.index(
        tqdm(
            corpus(),
            desc="BM25",
            unit="docs",
        )
    )

    print("\nBM25 indexing complete.")

# ============================================================================
# LOAD BM25 INDEX
# ============================================================================

print("\nLoading BM25 index into memory...")

existing_index = pt.IndexFactory.of(
    constants.INDEX_PATH,
    memory=True,
)

print(
    f"Loaded BM25 index with "
    f"{existing_index.getCollectionStatistics().getNumberOfDocuments():,} docs"
)

# ============================================================================
# STEP 2 — TCT-COLBERT INDEX
# ============================================================================

print("\n" + "=" * 80)
print("2. TCT-COLBERT INDEX")
print("=" * 80)

TCT_INDEX_PATH = constants.TCT_INDEX_PATH

if os.path.exists(os.path.join(TCT_INDEX_PATH, "pt_meta.json")):
    print("TCT-ColBERT index already exists.")
    print(TCT_INDEX_PATH)

else:

    print("\nLoading TCT-ColBERT model...")

    model = TctColBert(
        "castorini/tct_colbert-v2-hnp-msmarco",
        device=device,
        batch_size=DENSE_BATCH_SIZE,
        verbose=True,
    )

    if torch.cuda.is_available():

        model.model = model.model.half()

        try:
            model.model = torch.compile(model.model)
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    index_pipeline = model >> FlexIndex(TCT_INDEX_PATH)

    print("\nStarting TCT indexing...")

    with torch.inference_mode():

        index_pipeline.index(
            tqdm(
                corpus(),
                desc="TCT-ColBERT",
                unit="docs",
            )
        )

    print("\nTCT-ColBERT indexing complete.")

# ============================================================================
# STEP 3 — TASB INDEX
# ============================================================================

print("\n" + "=" * 80)
print("3. TASB INDEX")
print("=" * 80)

TASB_INDEX_PATH = constants.TASB_INDEX_PATH

if os.path.exists(os.path.join(TASB_INDEX_PATH, "pt_meta.json")):
    print("TasB index already exists.")
    print(TASB_INDEX_PATH)

else:

    print("\nLoading TasB model...")

    model = TasB.dot(
        batch_size=DENSE_BATCH_SIZE,
        device=device,
    )

    if torch.cuda.is_available():

        model.model = model.model.half()

        try:
            model.model = torch.compile(model.model)
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    idx_pipeline = model >> FlexIndex(TASB_INDEX_PATH)

    print("\nStarting TasB indexing...")

    with torch.inference_mode():

        idx_pipeline.index(
            tqdm(
                corpus(),
                desc="TasB",
                unit="docs",
            )
        )

    print("\nTasB indexing complete.")

# ============================================================================
# TEXT LOADER FOR LAFF
# ============================================================================

print("\nPreparing text loader for LAFF...")

if docs_df is not None:

    # FASTEST OPTION
    text_dict = dict(
        zip(
            docs_df["docno"],
            docs_df["text"],
        )
    )

    text_loader = pt.apply.generic(
        lambda df: df.assign(
            text=df["docno"].map(text_dict)
        )
    )

else:

    # fallback
    try:
        text_loader = pt.text.get_text(
            existing_index,
            "text",
        )
    except Exception:
        if DATASET_NAME == "msmarco-passage":
            dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{SPLIT}/judged")
            text_loader = pt.text.get_text(dataset, "text")
        elif DATASET_NAME == "dpr-w100":
            dataset = pt.get_dataset(f"irds:dpr-w100/trivia-qa/{SPLIT}")
            text_loader = pt.text.get_text(dataset, "text")
        else:
            raise

# ============================================================================
# ARTIFACT HELPER — resolve HF slugs to local paths (corpus graphs only)
# ============================================================================

def _is_hf_slug(path: str) -> bool:
    """True when path looks like an HF repo slug (owner/repo), not a local path."""
    return (
        bool(path)
        and not path.startswith((".", "/", "~"))
        and not os.path.exists(path)
        and "/" in path
    )


def _local_for_hf(hf_slug: str) -> str:
    repo_name = hf_slug.split("/")[-1]
    return os.path.join("indices", repo_name)


def _ensure_local(path: str) -> str:
    """If path is an HF slug, download it to ./indices/<name>/ and return
    the local path. If already local, return unchanged."""
    if not path:
        return path
    if not _is_hf_slug(path):
        return path
    local_path = _local_for_hf(path)
    pt_meta = os.path.join(local_path, "pt_meta.json")
    if os.path.exists(pt_meta):
        print(f"  (already local: {local_path})")
        return local_path
    print(f"  Downloading HF artifact '{path}' → {local_path} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=path,
        repo_type="dataset",
        local_dir=local_path,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )
    print(f"  Download complete: {local_path}")
    return local_path


# ============================================================================
# STEP 4 — BM25 CORPUS GRAPH (k=16)
# ============================================================================

print("\n" + "=" * 80)
print("4. BM25 CORPUS GRAPH (k=16)")
print("=" * 80)

GRAPH_K16_PATH = constants.CORPUS_GRAPH_PATH
GRAPH_K16_PATH = _ensure_local(GRAPH_K16_PATH) if GRAPH_K16_PATH else GRAPH_K16_PATH

if not GRAPH_K16_PATH or os.path.exists(GRAPH_K16_PATH):

    print("k=16 graph already exists (or path empty — skipping).")
    if GRAPH_K16_PATH:
        print(GRAPH_K16_PATH)

else:

    K = 16

    safe_bm25 = (
        pt.rewrite.tokenise()
        >>
        pt.terrier.Retriever(
            existing_index,
            wmodel="BM25",
            num_results=K + 1,
            threads=os.cpu_count(),
        )
    )

    print("\nBuilding k=16 CorpusGraph...")

    CorpusGraph.from_retriever(
        safe_bm25,
        corpus(),
        GRAPH_K16_PATH,
        k=K,
    )

    print("\nk=16 CorpusGraph complete.")

# ============================================================================
# STEP 5 — BM25 CORPUS GRAPH (k=128)
# ============================================================================

print("\n" + "=" * 80)
print("5. BM25 CORPUS GRAPH (k=128)")
print("=" * 80)

LAFF_BASE_PATH = constants.LAFF_BASE_GRAPH_PATH
LAFF_BASE_PATH = _ensure_local(LAFF_BASE_PATH) if LAFF_BASE_PATH else LAFF_BASE_PATH

if not LAFF_BASE_PATH or os.path.exists(LAFF_BASE_PATH):

    print("k=128 base graph already exists (or path empty — skipping).")
    if LAFF_BASE_PATH:
        print(LAFF_BASE_PATH)

else:

    K = 128

    safe_bm25 = (
        pt.rewrite.tokenise()
        >>
        pt.terrier.Retriever(
            existing_index,
            wmodel="BM25",
            num_results=K + 1,
            threads=os.cpu_count(),
        )
    )

    print("\nBuilding k=128 CorpusGraph...")

    CorpusGraph.from_retriever(
        safe_bm25,
        corpus(),
        LAFF_BASE_PATH,
        k=K,
    )

    print("\nk=128 CorpusGraph complete.")

# # ============================================================================
# # STEP 6 — LAFF GRAPH
# # ============================================================================

# print("\n" + "=" * 80)
# print("6. LAFF GRAPH")
# print("=" * 80)

# LAFF_GRAPH_PATH = constants.LAFF_GRAPH_PATH

# if os.path.exists(LAFF_GRAPH_PATH):

#     print("LAFF graph already exists.")
#     print(LAFF_GRAPH_PATH)

# else:

#     print("\nLoading LAFF model...")

#     laff_model = Laff(
#         device=device,
#         batch_size=LAFF_BATCH_SIZE,
#         max_length=512,
#     )

#     # ------------------------------------------------------------------------
#     # FP16 + COMPILE
#     # ------------------------------------------------------------------------

#     if (
#         torch.cuda.is_available()
#         and hasattr(laff_model, "model")
#     ):

#         try:
#             laff_model.model = laff_model.model.half()
#         except Exception:
#             pass

#         try:
#             laff_model.model = torch.compile(
#                 laff_model.model
#             )
#             print("torch.compile enabled.")
#         except Exception as e:
#             print(f"torch.compile skipped: {e}")

#     print("\nApplying LAFF to graph...")

#     with torch.inference_mode():

#         laff_model.apply_to_graph(
#             CorpusGraph.load(LAFF_BASE_PATH),
#             text_loader,
#             LAFF_GRAPH_PATH,
#             verbose=True,
#         )

#     print("\nLAFF graph complete.")

# ============================================================================
# FINISHED
# ============================================================================

print("\n" + "=" * 80)
print("ALL INDICES + CORPUS GRAPHS READY")
print("=" * 80)

print(f"\nDataset: {DATASET_NAME}")
print(f"Split: {SPLIT}")

print("\nArtifacts:")
print(f"- BM25 Index:        {BM25_INDEX_DIR}")
print(f"- TCT Index:         {TCT_INDEX_PATH}")
print(f"- TasB Index:        {TASB_INDEX_PATH}")
print(f"- Graph k=16:        {GRAPH_K16_PATH}")
print(f"- Graph k=128:       {LAFF_BASE_PATH}")
# print(f"- LAFF Graph:        {LAFF_GRAPH_PATH}")

print("\nPipeline complete.")
