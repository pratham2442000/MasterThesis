import numpy as np
from collections import defaultdict, Counter


def proportional_ranked_choice(rankings, k, method='borda', weight=None):
    """
    Select top k documents using proportional-ranked choice voting.
    """

    # Flatten rankings and get all unique documents
    all_docs = set()
    for ranking in rankings:
        all_docs.update(ranking)
    all_docs = list(all_docs)

    if method == 'borda':
        return borda_count(rankings, k, all_docs, weight=weight)
    else:
        raise ValueError("Method must be 'borda'.")


def borda_count(rankings, k, all_docs, weight=None):
    """Borda count method - assign points based on rank positions."""
    if weight is None:
        weight = lambda x : 1 # Uniform weight

    doc_scores = defaultdict(float)
    n_queries = len(rankings)

    for query_ranking in rankings:
        max_rank = len(query_ranking)
        for rank, doc in enumerate(query_ranking):
            # Borda score: higher ranks get more points
            score = (max_rank - rank) * weight(max_rank - rank)
            doc_scores[doc] += score

    # Normalize by number of queries
    for doc in doc_scores:
        doc_scores[doc] /= n_queries
    print(doc_scores)

    # Select top k documents
    top_docs = sorted(all_docs, key=lambda x: doc_scores[x], reverse=True)[:k]
    return top_docs



# Example usage
if __name__ == "__main__":
    # Example rankings from 3 queries
    rankings = [
        ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6'],  # Query 1 results
        ['doc2', 'doc3', 'doc1', 'doc6', 'doc7'],  # Query 2 results
        ['doc3', 'doc1', 'doc4', 'doc2', 'doc8']]   # Query 3 results

    k = 3  # Number of top documents to select
    selected_docs = proportional_ranked_choice(rankings, k, method='borda')
    selected_docs_w = proportional_ranked_choice(rankings, k, method='borda', weight = lambda x: np.e**(x*0.1))
    print(selected_docs, selected_docs_w)