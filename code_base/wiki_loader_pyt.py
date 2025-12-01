import json
import os

import pandas as pd
import tqdm


# Assumes data_path/$split$.json and data_path/corpus.json
def wiki_loader_pyt(data_path: str, split: str) -> pd.DataFrame:
    dataset_path = os.path.join(data_path, split+str(".json"))
    corpus_path = os.path.join(data_path, "wiki_musique_corpus.json")

    print("Processing corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)

    docs_list = []

    for doc_id, snippet in tqdm.tqdm(corpus_data.items()):
        docs_list.append({
            "docno": str(doc_id),
            "title": snippet['title'],
            "text": snippet['text']
        })
    docs_df = pd.DataFrame(docs_list)

    print(f"Number of duplicates: {docs_df.duplicated(subset=['text']).sum()}")

    docs_df = docs_df.drop_duplicates(subset=['text']).reset_index(drop=True)

    # Create a mapping from title to docno(s) for qrels
    print("Building title to docno mapping...")
    title_to_docnos = {}
    for index, row in tqdm.tqdm(docs_df.iterrows()):
        t = row['title']
        d = row['docno']
        if t not in title_to_docnos:
            title_to_docnos[t] = []
        title_to_docnos[t].append(d)
    print(f"Created docs_df with {len(docs_df)} documents.")

    print(f"\nProcessing {split} set for topics and qrels...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        query_data = json.load(f)

    topics_list = []
    qrels_list = []
    for data in tqdm.tqdm(query_data):
        query_id = data["_id"]

        topics_list.append({
            "qid": query_id,
            "query": data["question"],
            "type": data.get("type", "unknown")
        })

        for fact in data["context"]:
            doc_title = fact[0]
            if doc_title in title_to_docnos:
                for docno in title_to_docnos[doc_title]:
                    qrels_list.append({
                        "qid": query_id,
                        "docno": docno,
                        "label": 1
                    })

    topics_df = pd.DataFrame(topics_list)
    qrels_df = pd.DataFrame(qrels_list)

    docs_df = docs_df.dropna(subset=['docno', 'text'])
    docs_df['docno'] = docs_df['docno'].astype(str)
    docs_df = docs_df[docs_df['text'].str.strip() != '']
    print("Document DataFrame info after cleanup:")
    print(docs_df.info())
    print("/n/n")
    print(f"Created topics_df with {len(topics_df)} queries.")
    print(f"Created qrels_df with {len(qrels_df)} relevance judgments.")

    print("/n/n/n")


    return docs_df, topics_df, qrels_df



if __name__ == "__main__":

    DEV_FILE = "/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA"
    CORPUS_FILE = 'corpus.json'


    docs_df, topics_df, qrels_df = wiki_loader_pyt(data_path=DEV_FILE, split='dev')


    print("Courpus DataFrame")
    print(docs_df.head())
    print(docs_df.info())

    print("Topics DataFrame")
    print(topics_df.head())
    print(topics_df.info())

    print("Qrels DataFrame")
    print(qrels_df.head())
    print(qrels_df.info())


#dev set
#
# {
#   "_id": "string",
#   "type": "string",
#   "question": "string",
#   "context": "array of arrays of strings",
#   "supporting_facts": "array of arrays (mixed: string and integer)",
#   "evidences": "array of arrays of strings",
#   "answer": "string"
# }


#corpus
#
# {
#   "0": { // Numerical string key
#     "title": "string", // Title of the document
#     "text": "string"   // Content of the document snippet
#   },
#   "1": { ... },
#   // ... more snippets
# }
