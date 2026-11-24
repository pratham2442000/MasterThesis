"""
Generate LaTeX tables for each experiment,
including only the columns that appear in the plots of Results Plots.ipynb.
"""

import os

import pandas as pd

from code_base.experiment_utils import find_experiment_files

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'results', 'latex_tables')
os.makedirs(OUT_DIR, exist_ok=True)


def save_latex(df, name, caption, label):
    """Write a LaTeX table to file and print it."""
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
    )
    path = os.path.join(OUT_DIR, f'{name}.tex')
    with open(path, 'w') as f:
        f.write(latex)
    print(f'\n{"=" * 60}')
    print(f'  {name}  →  {path}')
    print(f'{"=" * 60}')
    print(latex)


# =============================================================
# Experiment 1
# =============================================================
files = find_experiment_files('experiment_1', os.path.join(BASE, 'results/exp1'))
df = pd.read_csv(files['csv'])

cols = ['c', 'BM25_R@10', 'Pipeline_R@10',
        'BM25_nDCG@10', 'Pipeline_nDCG@10']
df_exp1 = df[[c for c in cols if c in df.columns]]

save_latex(df_exp1, 'exp1_table',
           caption='Experiment 1: Effect of Initial Cutoff (c) on Retrieval Metrics',
           label='tab:exp1')

# =============================================================
# Experiment 2  (Merge-then-Rerank)
# =============================================================
files = find_experiment_files('experiment_2', os.path.join(BASE, 'results/exp2'))
df = pd.read_csv(files['csv'])

systems_2 = ['BM25', 'BM25_MonoT5', 'Pipeline_Borda', 'Pipeline_BordaWeighted',
             'Pipeline_CombMAX', 'Pipeline_Dedup', 'Pipeline_DedupCutoff']
cols = ['n']
for s in systems_2:
    for m in ['R@10', 'nDCG@10']:
        c = f'{s}_{m}'
        if c in df.columns:
            cols.append(c)

df_exp2 = df[cols]
save_latex(df_exp2, 'exp2_table',
           caption='Experiment 2: Merge-then-Rerank — Fusion Methods vs Number of Rewrites',
           label='tab:exp2')

# =============================================================
# Experiment 2 Noise Analysis
# =============================================================
files = find_experiment_files('experiment_2_noise_analysis',
                              os.path.join(BASE, 'results/exp2_noise'))
df = pd.read_csv(files['csv'])

cols = ['n'] + [c for c in df.columns if c != 'n' and '_R@' in c]
df_noise = df[cols]

save_latex(df_noise, 'exp2_noise_table',
           caption='Experiment 2 Noise Analysis: Recall vs Number of Rewrites',
           label='tab:exp2_noise')

# =============================================================
# Experiment 2a  (Dense Retrieval)
# =============================================================
files = find_experiment_files('experiment_2a', os.path.join(BASE, 'results/exp2a'))
df = pd.read_csv(files['csv'])

systems_2a = ['Dense', 'Dense_MonoT5', 'Pipeline_Borda',
              'Pipeline_BordaWeighted', 'Pipeline_CombMAX']
cols = ['n']
for s in systems_2a:
    for m in ['R@10', 'nDCG@10']:
        c = f'{s}_{m}'
        if c in df.columns:
            cols.append(c)

df_exp2a = df[cols]
save_latex(df_exp2a, 'exp2a_table',
           caption='Experiment 2a: Dense Retrieval — Fusion Methods vs Number of Rewrites',
           label='tab:exp2a')

# =============================================================
# Experiment 2b  (Rerank-then-Merge)
# =============================================================
files = find_experiment_files('experiment_2b', os.path.join(BASE, 'results/exp2b'))
df = pd.read_csv(files['csv'])

systems_2b = ['BM25', 'BM25_MonoT5', 'Pipeline_Borda', 'Pipeline_BordaWeighted',
              'Pipeline_CombMAX', 'Pipeline_Dedup']
cols = ['n']
for s in systems_2b:
    for m in ['R@10', 'nDCG@10']:
        c = f'{s}_{m}'
        if c in df.columns:
            cols.append(c)

df_exp2b = df[cols]
save_latex(df_exp2b, 'exp2b_table',
           caption='Experiment 2b: Rerank-then-Merge — Fusion Methods vs Number of Rewrites',
           label='tab:exp2b')

# =============================================================
# Experiment 3
# =============================================================
files = find_experiment_files('experiment_3', os.path.join(BASE, 'results/exp3'))
df = pd.read_csv(files['csv'])

cols = ['cutoff', 'BM25_R@10', 'BM25_MonoT5_R@10', 'CombMAX_R@10', "CombMAX_RTM_R@10",
        'BM25_nDCG@10', 'BM25_MonoT5_nDCG@10', 'CombMAX_nDCG@10', "CombMAX_RTM_nDCG@10"]

print(df.columns)

df_exp3 = df[[c for c in cols if c in df.columns]]

save_latex(df_exp3, 'exp3_table',
           caption='Experiment 3: Effect of Initial Cutoff (c) on Retrieval Metrics',
           label='tab:exp3')

print(f'\n\nAll tables saved to: {OUT_DIR}')
