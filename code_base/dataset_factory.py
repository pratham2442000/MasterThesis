from code_base.wiki_loader_pyt import wiki_loader_pyt
import pyterrier as pt



def get_dataset(name: str, SPLIT: str):
    if name == "wiki_musique":
        DATA_PATH = "/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA"
        return wiki_loader_pyt(DATA_PATH, SPLIT)
    elif name == "msmarco-passage":
        dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{SPLIT}/judged')
        return dataset.get_corpus(), dataset.get_topics(), dataset.get_qrels()
    else:
        raise ValueError(f"Dataset {name} not recognized.")
