import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel


def download_model(model_name, output_dir):
    print(f"Downloading {model_name}...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 1. Save Tokenizer (Works for everything)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)

        # 2. Try saving as Seq2Seq (for T5/MonoT5)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except (OSError, ValueError):
            # 3. If that fails, load as a standard Model (for BERT/ColBERT)
            print(f"  > Not a Seq2Seq model. Loading as standard AutoModel...")
            model = AutoModel.from_pretrained(model_name)

        model.save_pretrained(output_dir)
        print(f"Successfully downloaded {model_name} to {output_dir}\n")

    except Exception as e:
        print(f"CRITICAL ERROR downloading {model_name}: {e}")


if __name__ == "__main__":
    base_path = "/scratch/pjohari/MasterThesis/models"

    models_to_download = [
        "castorini/monot5-base-msmarco",  # Works with Seq2Seq
        "t5-base",  # Works with Seq2Seq
        "castorini/tct_colbert-msmarco"  # REQUIRES AutoModel fallback
    ]

    for model_name in models_to_download:
        output_dir = os.path.join(base_path, model_name)
        download_model(model_name, output_dir)