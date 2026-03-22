"""
Experiment 2 Setup Script
==========================
This script runs the preprep_dataset function to create the dataset
of rewritten queries. This must be run before experiment_2.py.

The dataset will be saved to a CSV file that experiment_2.py will read from.
"""

import os
import sys

# Add parent directory to path to allow importing constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import constants
from code_base.dataset_factory import get_dataset
from code_base.llm_q import preprep_dataset


def main():
    """
    Main function to run experiment 2 setup.
    Loads dataset and creates rewritten queries using LLM.
    """
    print("=" * 60)
    print("Experiment 2 Setup: Creating Query Rewrite Dataset")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {constants.DATASET_NAME}")
    print(f"  Data Split: {constants.DATA_SPLIT}")
    print(f"  Data Path: {constants.DATA_PATH}")
    n_rewrites = constants.LLM_REWRITES_PER_QUERY
    print(f"  Rewrites per Query: {n_rewrites}")
    topic_limit = constants.EXPERIMENT_2_TOPIC_LIMIT
    print(f"  Topic Limit: {topic_limit}")
    api_permission = constants.REQUIRE_API_PERMISSION
    print(f"  API Permission Required: {api_permission}")

    type_distribution = constants.EXPERIMENT_2_TYPE_DISTRIBUTION
    print(f"  Type Distribution: {type_distribution}")
    print("\n" + "=" * 60 + "\n")

    # Load dataset
    print("Loading dataset...")
    try:
        corpus_df, topics_df, qrels_df = get_dataset(
            constants.DATASET_NAME, constants.DATA_SPLIT
        )
        print(f"Loaded dataset: {len(topics_df)} topics")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Limit topics if specified
    sample_limit = None
    if constants.EXPERIMENT_2_TOPIC_LIMIT > 0:
        sample_limit = constants.EXPERIMENT_2_TOPIC_LIMIT
        print(f"Set sample limit to {sample_limit}")

    # Prepare questions list: (question, q_id, type) tuples
    # The 'type' column exists for all datasets (set to 'unknown' for MSMARCO)
    questions = list(zip(topics_df["query"], topics_df["qid"], topics_df["type"]))

    print(f"\nPreparing to rewrite {len(questions)} queries...")
    print("This will make API calls to OpenAI.")

    # Run preprep_dataset with live=True to make actual API calls
    # type_distribution is None for MSMARCO, which skips type-based sampling
    try:
        df = preprep_dataset(
            name=constants.DATASET_NAME,
            questions=questions,
            split=constants.DATA_SPLIT,
            path_dt=constants.DATA_PATH,
            live=True,  # Set to False for testing without API calls
            type_distribution=type_distribution,
            sample_limit=sample_limit
        )

        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        n_rewrites = constants.LLM_REWRITES_PER_QUERY
        print(f"\nDataset saved with {len(df)} queries")
        print(f"Each query has {n_rewrites} rewrites")
        print("\nYou can now run experiment_2.py")

    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\nError during setup: {e}")
        raise


if __name__ == "__main__":
    main()
