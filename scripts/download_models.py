import os
import sys

# Add parent to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import snapshot_download
import constants


def main():
    print(f"Models will be downloaded to cache dir: {constants.MODELS_CACHE_DIR}")
    os.makedirs(constants.MODELS_CACHE_DIR, exist_ok=True)

    models_to_download = [
        constants.RAG_MODEL_NAME,  # Generation
        constants.RERANKER_MODEL_NAME,  # Reranking
        constants.DENSE_MODEL_NAME,  # Dense Retrieval
        constants.CROSS_ENCODER_MODEL_NAME  # Cross Encoder
    ]

    for model_id in models_to_download:
        if not model_id:
            continue
        print(f"\n--- Downloading {model_id} ---")
        try:
            # Downlod to the specific local directory to ensure fully offline setup
            local_dir = os.path.join(constants.MODELS_CACHE_DIR, model_id.split('/')[-1])
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # Better for Windows offline
                resume_download=True
            )
            print(f"Successfully downloaded {model_id} to {local_dir}")
        except Exception as e:
            print(f"Error downloading {model_id}: {e}")


if __name__ == "__main__":
    main()
