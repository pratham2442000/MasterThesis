"""
Download RAG generation model for offline use on DelftBlue.

Follows the same pattern as download_models.py. Downloads the model and
tokenizer so they can be loaded without internet on compute nodes.
"""
import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants


def download_causal_lm(model_name: str, output_dir: str):
    """Download a CausalLM model + tokenizer to a local directory."""
    print(f"Downloading {model_name} -> {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        print(f"  ✓ Tokenizer saved")

        # 2. Model (auto dtype to respect bf16/fp16 metadata)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto"
        )
        model.save_pretrained(output_dir)
        print(f"  ✓ Model saved")

        print(f"Successfully downloaded {model_name}\n")

    except Exception as e:
        print(f"CRITICAL ERROR downloading {model_name}: {e}")


if __name__ == "__main__":
    # Base path for model storage
    base_path = os.environ.get("MODELS_CACHE_DIR", os.path.abspath(constants.MODELS_CACHE_DIR))

    # Download the RAG generation model from constants
    model_name = constants.RAG_MODEL_NAME
    # Create a clean directory name from the model identifier
    safe_name = model_name.replace("/", "_")
    output_dir = os.path.join(base_path, safe_name)

    download_causal_lm(model_name, output_dir)
