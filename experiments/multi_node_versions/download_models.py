import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def download_model(model_name, output_dir):
    """
    Downloads a model and tokenizer from Hugging Face and saves it locally.
    """
    print(f"Downloading {model_name}...")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory already exists: {output_dir}")

    try:
        # Download and save tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer saved.")

        # Download and save model
        print("Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(output_dir)
        print("Model saved.")
        
        print(f"Successfully downloaded {model_name} to {output_dir}")

    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    base_path = "/scratch/pjohari/MasterThesis/models"
    
    models_to_download = [
        "castorini/monot5-base-msmarco",
        "t5-base"
    ]

    for model_name in models_to_download:
        output_dir = os.path.join(base_path, model_name)
        download_model(model_name, output_dir)
