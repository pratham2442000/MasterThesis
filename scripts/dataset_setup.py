"""
Unified Dataset Setup Script
==============================
Runs all setup steps needed before experiments can execute:
  1. Build BM25 index (needed for experiments 1, 2, 2b, 3)
  2. Generate query rewrites via LLM (needed for experiments 2, 2a, 2b, 3)
  3. Build dense index (needed for experiment 2a, optional)

Usage:
    python scripts/dataset_setup.py                  # run all steps
    python scripts/dataset_setup.py --steps index     # BM25 index only
    python scripts/dataset_setup.py --steps rewrites      # query rewrites only
    python scripts/dataset_setup.py --steps best_rewrite  # best rewrite selection only
    python scripts/dataset_setup.py --steps dense         # dense index only
    python scripts/dataset_setup.py --steps index rewrites  # multiple steps
    python scripts/dataset_setup.py --skip-dense      # all except dense

All configuration is read from constants.py. To switch datasets,
change constants.DATASET_NAME and re-run this script.
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyterrier as pt

import constants
from code_base.dataset_factory import get_dataset


def step_build_bm25_index(force_rebuild: bool = False):
    """Step 1: Build BM25 index from the dataset corpus.

    For MSMARCO-passage, first tries to use PyTerrier's pre-built index
    via ``pt.get_dataset('msmarco_passage').get_index('terrier_stemmed')``.
    Falls back to building from corpus if the pre-built index is unavailable.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Building BM25 Index")
    print("=" * 60)

    INDEX_DIR = os.path.dirname(constants.INDEX_PATH)
    print(f"  Dataset: {constants.DATASET_NAME}")
    print(f"  Index Directory: {INDEX_DIR}")

    # Check if index already exists at the configured path
    if os.path.exists(constants.INDEX_PATH) and not force_rebuild:
        print(f"  ✓ Index already exists at {constants.INDEX_PATH}")
        try:
            index = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
            n_docs = index.getCollectionStatistics().getNumberOfDocuments()
            print(f"  ✓ Index valid — {n_docs} documents")
            print("  Skipping index creation. Use --force-rebuild to rebuild.")
            return
        except Exception as e:
            print(f"  ✗ Index exists but is corrupt: {e}")
            print("  Will rebuild.")

    if constants.DATASET_NAME == "msmarco-passage" and not force_rebuild:
        # MSMARCO: Try PyTerrier's pre-built index first (auto-downloads)
        print("  Trying PyTerrier pre-built index for msmarco_passage...")
        try:
            dataset = pt.get_dataset("msmarco_passage")
            index_ref = dataset.get_index("terrier_stemmed")
            print(f"  ✓ Pre-built index obtained: {index_ref}")

            # Verify index works
            index = pt.IndexFactory.of(index_ref, memory=True)
            n_docs = index.getCollectionStatistics().getNumberOfDocuments()
            print(f"  ✓ Index has {n_docs} documents")

            # Copy/symlink to configured path so experiments find it
            import shutil
            # get_index() returns the index directory path itself
            src_dir = str(index_ref)
            os.makedirs(INDEX_DIR, exist_ok=True)

            # If pre-built index is at a different location, copy files
            if os.path.abspath(src_dir) != os.path.abspath(INDEX_DIR):
                print(f"  Copying pre-built index to {INDEX_DIR}...")
                for f in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, f)
                    dst_file = os.path.join(INDEX_DIR, f)
                    # Only copy files, skip subdirectories
                    if os.path.isfile(src_file) and not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
                print(f"  ✓ Index files copied to {INDEX_DIR}")
            else:
                print(f"  ✓ Index already at configured path")

            print()
            print("  ⚠  WARNING: Pre-built index does NOT have meta.reverse.keys=docno.")
            print("     Experiment 7 (ReformIR) requires reverse docno lookup for RM3.")
            print("     To run experiment_7.py, re-index with:")
            print("       python scripts/dataset_setup.py --steps index --force-rebuild")
            print()
            return

        except Exception as e:
            print(f"  ✗ Pre-built index failed: {e}")
            print("  Falling back to building index from corpus...")

        # Fallback: build from corpus via IterDictIndexer
        print("  Loading MSMARCO corpus via ir_datasets (this may take a while)...")
        dataset = pt.get_dataset('irds:msmarco-passage')
        os.makedirs(INDEX_DIR, exist_ok=True)
        indexer = pt.IterDictIndexer(
            INDEX_DIR, overwrite=True,
            meta={'docno': 256, 'text': 4096},
            meta_reverse=['docno'],
        )
        index_ref = indexer.index(
            dataset.get_corpus_iter(),
        )

    elif constants.DATASET_NAME == "dpr-w100":
        # DPR-W100: ~21M Wikipedia passages, build from ir_datasets
        print("  Loading DPR-W100 corpus via ir_datasets (~21M passages)...")
        print("  This will take a while. Text metadata is stored for RAG.")
        dataset = pt.get_dataset('irds:dpr-w100')
        os.makedirs(INDEX_DIR, exist_ok=True)
        indexer = pt.IterDictIndexer(
            INDEX_DIR, overwrite=True,
            meta={'docno': 256},
            meta_reverse=['docno'],
        )
        index_ref = indexer.index(
            dataset.get_corpus_iter(),
        )
    else:
        # Custom datasets: load corpus into DataFrame first
        print("  Loading corpus...")
        docs_df, _, _ = get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)

        if docs_df is None:
            print("  ERROR: No corpus available for indexing.")
            return

        # Clean documents
        docs_df = docs_df.dropna(subset=['docno', 'text'])
        docs_df['docno'] = docs_df['docno'].astype(str)
        docs_df = docs_df[docs_df['text'].str.strip() != '']
        print(f"  Documents to index: {len(docs_df)}")

        os.makedirs(INDEX_DIR, exist_ok=True)
        indexer = pt.IterDictIndexer(
            INDEX_DIR, overwrite=True,
            meta={'docno': 256, 'text': 4096},
            meta_reverse=['docno'],
        )
        index_ref = indexer.index(docs_df.to_dict(orient='records'))

    print(f"  ✓ BM25 index created: {index_ref}")


def step_generate_rewrites():
    """Step 2: Generate query rewrites via LLM."""
    print("\n" + "=" * 60)
    print("STEP 2: Generating Query Rewrites via LLM")
    print("=" * 60)

    from code_base.llm_q import preprep_dataset

    print(f"  Dataset: {constants.DATASET_NAME}")
    print(f"  Split: {constants.DATA_SPLIT}")
    print(f"  Rewrites per query: {constants.LLM_REWRITES_PER_QUERY}")
    print(f"  Topic limit: {constants.EXPERIMENT_2_TOPIC_LIMIT}")
    print(f"  Type distribution: {constants.EXPERIMENT_2_TYPE_DISTRIBUTION}")

    # Load dataset
    print("  Loading topics...")
    _, topics_df, _ = get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)
    print(f"  Loaded {len(topics_df)} topics")

    # Determine sample limit
    sample_limit = None
    if constants.EXPERIMENT_2_TOPIC_LIMIT > 0:
        sample_limit = constants.EXPERIMENT_2_TOPIC_LIMIT
        print(f"  Sample limit: {sample_limit}")

    # Build questions list
    questions = list(zip(topics_df["query"], topics_df["qid"], topics_df["type"]))
    # Check if rewrites already exist
    n_rewrites = constants.LLM_REWRITES_PER_QUERY
    filename = f"{constants.DATASET_NAME}_rewritten_queries_dataset_{n_rewrites}_{constants.DATA_SPLIT}.csv"
    filepath = os.path.join(constants.DATA_PATH, filename)

    if os.path.exists(filepath):
        print(f"  Existing rewrites found at: {filepath}")
        print("  Will skip already-processed queries (incremental mode).")

    # Run rewriting
    df = preprep_dataset(
        name=constants.DATASET_NAME,
        questions=questions,
        split=constants.DATA_SPLIT,
        path_dt=constants.DATA_PATH,
        live=True,
        type_distribution=constants.EXPERIMENT_2_TYPE_DISTRIBUTION,
        sample_limit=sample_limit
    )

    print(f"  Rewrites saved: {len(df)} queries with {n_rewrites} rewrites each")


def step_select_best_rewrites():
    """Step 2b: Select the best rewrite per N value via LLM."""
    print("\n" + "=" * 60)
    print("STEP 2b: Selecting Best Rewrites via LLM")
    print("=" * 60)

    from code_base.llm_best_rewrite import preprep_best_rewrites

    print(f"  Dataset: {constants.DATASET_NAME}")
    print(f"  Split: {constants.DATA_SPLIT}")
    print(f"  N values: {constants.EXPERIMENT_2_N_REWRITES}")

    df = preprep_best_rewrites(
        name=constants.DATASET_NAME,
        split=constants.DATA_SPLIT,
        path_dt=constants.DATA_PATH,
        live=True,
        n_values=constants.EXPERIMENT_2_N_REWRITES,
    )

    print(f"  Best rewrites saved: {len(df)} queries")


def step_build_dense_index():
    """Step 3: Build dense index for experiment 2a."""
    print("\n" + "=" * 60)
    print("STEP 3: Building Dense Index (TCT-ColBERT)")
    print("=" * 60)

    try:
        import pyterrier_dr as pt_dr
    except ImportError:
        print("  ERROR: pyterrier_dr not installed. Run: pip install pyterrier-dr")
        return

    model_name = constants.DENSE_MODEL_NAME
    index_path = constants.DENSE_INDEX_PATH

    print(f"  Model: {model_name}")
    print(f"  Index Path: {index_path}")

    # Check if already exists
    if os.path.exists(os.path.join(index_path, "index.dpr")):
        print(f"  Warning: Dense index already exists at {index_path}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("  Skipping dense index creation.")
            return

    # Load corpus
    if constants.DATASET_NAME == "msmarco-passage":
        print("  Loading MSMARCO corpus via ir_datasets...")
        dataset = pt.get_dataset('irds:msmarco-passage')
        model = pt_dr.TctColBert(model_name, verbose=True)
        index = pt_dr.FlexIndex(index_path)
        pipeline = model.doc_encoder() >> index.indexer(mode="overwrite")
        pipeline.index(dataset.get_corpus_iter())
    elif constants.DATASET_NAME == "dpr-w100":
        print("  Loading DPR-W100 corpus via ir_datasets (~21M passages)...")
        dataset = pt.get_dataset('irds:dpr-w100')
        model = pt_dr.TctColBert(model_name, verbose=True)
        index = pt_dr.FlexIndex(index_path)
        pipeline = model.doc_encoder() >> index.indexer(mode="overwrite")
        pipeline.index(dataset.get_corpus_iter())
    else:
        print("  Loading corpus...")
        docs_df, _, _ = get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)

        if docs_df is None:
            print("  ERROR: No corpus available for dense indexing.")
            return

        model = pt_dr.TctColBert(model_name, verbose=True)
        index = pt_dr.FlexIndex(index_path)
        pipeline = model.doc_encoder() >> index.indexer(mode="overwrite")
        pipeline.index(docs_df.to_dict(orient='records'))

    print("  Dense index created successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified dataset setup for all experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/dataset_setup.py                     # all steps
  python scripts/dataset_setup.py --steps index       # BM25 index only
  python scripts/dataset_setup.py --steps rewrites    # rewrites only
  python scripts/dataset_setup.py --skip-dense        # index + rewrites
        """
    )
    parser.add_argument(
        '--steps', nargs='+',
        choices=['index', 'rewrites', 'best_rewrite', 'dense'],
        default=None,
        help='Specific steps to run (default: all)'
    )
    parser.add_argument(
        '--skip-dense', action='store_true',
        help='Run all steps except dense indexing'
    )
    parser.add_argument(
        '--force-rebuild', action='store_true',
        help='Force rebuild BM25 index with reverse docno keys (needed for experiment_7/RM3)'
    )

    args = parser.parse_args()

    if not pt.started():
        pt.init()

    # Determine which steps to run
    if args.steps:
        steps = set(args.steps)
    elif args.skip_dense:
        steps = {'index', 'rewrites', 'best_rewrite'}
    else:
        steps = {'index', 'rewrites', 'best_rewrite', 'dense'}

    print("=" * 60)
    print("Unified Dataset Setup")
    print("=" * 60)
    print(f"  Dataset: {constants.DATASET_NAME}")
    print(f"  Split: {constants.DATA_SPLIT}")
    print(f"  Data Path: {constants.DATA_PATH}")
    print(f"  Steps: {', '.join(sorted(steps))}")
    print("=" * 60)

    # Execute steps in order
    if 'index' in steps:
        step_build_bm25_index(force_rebuild=args.force_rebuild)

    if 'rewrites' in steps:
        step_generate_rewrites()

    if 'best_rewrite' in steps:
        step_select_best_rewrites()

    if 'dense' in steps:
        step_build_dense_index()

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nDataset '{constants.DATASET_NAME}' is ready for experiments.")
    if 'index' in steps:
        print(f"  BM25 Index: {constants.INDEX_PATH}")
    if 'rewrites' in steps:
        n = constants.LLM_REWRITES_PER_QUERY
        fname = f"{constants.DATASET_NAME}_rewritten_queries_dataset_{n}_{constants.DATA_SPLIT}.csv"
        print(f"  Rewrites: {os.path.join(constants.DATA_PATH, fname)}")
    if 'best_rewrite' in steps:
        fname = f"{constants.DATASET_NAME}_best_rewrite_{constants.DATA_SPLIT}.csv"
        print(f"  Best Rewrites: {os.path.join(constants.DATA_PATH, fname)}")
    if 'dense' in steps:
        print(f"  Dense Index: {constants.DENSE_INDEX_PATH}")


if __name__ == "__main__":
    main()
