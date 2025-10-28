import pyterrier as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

if not pt.started():
    pt.init()

from pyterrier_t5 import MonoT5ReRanker
from pyterrier.measures import *


from code_base.wiki_loader_pyt import wiki_loader_pyt


# Data Configuration
DATA_PATH = "/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA"
DATA_SPLIT = "test"

# PyTerrier Index Configuration
INDEX_PATH = "./index/wiki_musique_index/data.properties"  # Path to the Terrier index metadata

# Experiment Configuration
# Limits the number of topics (queries) used for the experiment
TOPIC_LIMIT = 1000

# Initial Cutoff Values (c) for BM25 Retrieval
# Defines the range of 'c' to experiment with.
# Example: [10, 50, 100, 200, 500, 1000]
C_VALUES = [int(i) for i in np.linspace(10, 100, 10, dtype=int)] # e.g., [10, 55, 100]
T5_BATCH_SIZE = 16
EXPERIMENT_BATCH_SIZE = 20
EVAL_METRICS = [R@10, NDCG@10]



print("Loading data...")
docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, DATA_SPLIT)

def preprocess_query(query):
    """Simple query cleaning: remove punctuation/special chars and lowercases."""
    query = ''.join(e for e in query if e.isalnum() or e.isspace())
    return query.lower()

topics_df['query'] = topics_df['query'].apply(preprocess_query)

topics = topics_df[:TOPIC_LIMIT]
index_ref = pt.IndexFactory.of(INDEX_PATH)

mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)


results = {
    'c': C_VALUES,
    'BM25_R@10': [],
    'Pipeline_R@10': [],
    'BM25_NDCG@10': [],
    'Pipeline_NDCG@10': []
}

print(f"\nStarting experiment for initial cutoff (c) in: {C_VALUES}")





for c in C_VALUES:
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
    bm25_retrieved = bm25 % c
    pipeline = bm25_retrieved >> pt.text.get_text(index_ref, "text") >> mono_t5_reranker

    experiment_results = pt.Experiment(
        [bm25_retrieved, pipeline],
        topics,
        qrels_df,
        eval_metrics=EVAL_METRICS,
        names=["BM25", "BM25+MonoT5"],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
    )

    results['BM25_R@10'].append(float(experiment_results['R@10'][0]))
    results['Pipeline_R@10'].append(float(experiment_results['R@10'][1]))
    results['BM25_NDCG@10'].append(float(experiment_results['nDCG@10'][0]))
    results['Pipeline_NDCG@10'].append(float(experiment_results['nDCG@10'][1]))

    print(f"c={c}: BM25 R@10={results['BM25_R@10'][-1]:.4f}, Pipeline R@10={results['Pipeline_R@10'][-1]:.4f} | BM25 NDCG@10={results['BM25_NDCG@10'][-1]:.4f}, Pipeline NDCG@10={results['Pipeline_NDCG@10'][-1]:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_PATH+"/experiment_1_results.csv", index=False)
print("\nFinal Results DataFrame:")
print(results_df)


## CHART GENERATION


def plot_metrics(df, metric_suffix, title, ylabel, filename):
    """Generates and saves a plot for a given metric."""
    plt.figure(figsize=(10, 6))
    metric_bm25 = f'BM25_{metric_suffix}'
    metric_pipeline = f'Pipeline_{metric_suffix}'

    plt.plot(df['c'], df[metric_bm25], marker='o', label='BM25 Baseline')
    plt.plot(df['c'], df[metric_pipeline], marker='x', label='BM25 + MonoT5 Re-ranker')

    plt.title(title)
    plt.xlabel('Initial Cutoff (c) for BM25 Retrieval')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Use log scale for x-axis if there is a wide range of C_VALUES
    if len(df['c']) > 2 and (df['c'].max() / df['c'].min() > 5):
        plt.xscale('log')
        plt.xticks(df['c'], labels=[str(v) for v in df['c']])

    plt.tight_layout()
    plt.savefig(DATA_PATH+filename)
    print(f"Chart saved as {DATA_PATH+filename}")

# R@10 vs. Cutoff
plot_metrics(
    results_df,
    'R@10',
    'Effect of Initial Cutoff (c) on Recall (R@10)',
    'Recall (R@10)',
    '/cutoff_vs_R_at_10.png'
)


# NDCG@10 vs. Cutoff
plot_metrics(
    results_df,
    'NDCG@10',
    'Effect of Initial Cutoff (c) on NDCG@10',
    'NDCG@10 (Discounted Cumulative Gain)',
    '/cutoff_vs_NDCG_at_10.png'
)


## LINEAR REGRESSION ANALYSIS (R@10)


print("\n--- Linear Regression Analysis (log(c) vs R@10) ---")

# Use log(c) for regression as is common in IR analysis
log_c = np.log(results_df['c'])

def run_regression(x, y, label):
    """Performs and prints linear regression results."""
    # Perform linear regression: R@10 = slope * log(c) + intercept
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value**2

    print(f"\n{label}:")
    print(f"  Equation: R@10 = {slope:.4f} * log(c) + {intercept:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  P-value: {p_value:.4g}") # Lower p-value indicates stronger significance

    return slope, intercept

# Regression for BM25 R@10
slope_bm25, intercept_bm25 = run_regression(
    log_c, results_df['BM25_R@10'], "BM25 (Baseline) R@10"
)

# Regression for Pipeline R@10
slope_pipeline, intercept_pipeline = run_regression(
    log_c, results_df['Pipeline_R@10'], "Pipeline (BM25+MonoT5) R@10"
)

# Plot with Regression Lines (R@10)
plt.figure(figsize=(10, 6))

# Plot measured data points
plt.plot(results_df['c'], results_df['BM25_R@10'], marker='o', linestyle='', label='BM25 R@10 (Measured)')
plt.plot(results_df['c'], results_df['Pipeline_R@10'], marker='x', linestyle='', label='Pipeline R@10 (Measured)')

# Plot regression lines
c_values_for_plot = np.logspace(np.log10(results_df['c'].min()), np.log10(results_df['c'].max()), 50)
log_c_for_plot = np.log(c_values_for_plot)

plt.plot(
    c_values_for_plot,
    intercept_bm25 + slope_bm25 * log_c_for_plot,
    color='blue', linestyle='--', alpha=0.7,
    label='BM25 Regression Line'
)
plt.plot(
    c_values_for_plot,
    intercept_pipeline + slope_pipeline * log_c_for_plot,
    color='red', linestyle='--', alpha=0.7,
    label='Pipeline Regression Line'
)

plt.title('R@10 vs. Initial Cutoff (c) with Log(c) Linear Regression')
plt.xlabel('Initial Cutoff (c) for BM25 Retrieval (log scale)')
plt.ylabel('Recall (R@10)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xscale('log')
plt.xticks(results_df['c'], labels=[str(v) for v in results_df['c']])
plt.tight_layout()
plt.savefig('cutoff_R_at_10_log_regression.png')