"""
Experiment 2 vs 2b Comparison Plots
====================================
Exp 2  = Merge-then-Rerank
Exp 2b = Rerank-then-Merge

Generates:
  1. Per-metric line plots  (R@10, R@c, nDCG@10) overlaying both strategies
  2. Grouped bar charts at n=10
  3. Percentage-difference summary table
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code_base.experiment_utils import find_experiment_files

# ---------- Matplotlib defaults ----------
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.loc': 'lower right',
    'legend.framealpha': 0.5,
})

# ---------- Paths ----------
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'results', 'exp2_vs_2b')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------- Load data ----------
files_2  = find_experiment_files('experiment_2',  os.path.join(BASE, 'results/exp2'))
files_2b = find_experiment_files('experiment_2b', os.path.join(BASE, 'results/exp2b'))
df2  = pd.read_csv(files_2['csv'])
df2b = pd.read_csv(files_2b['csv'])

print("Exp 2  columns:", list(df2.columns))
print("Exp 2b columns:", list(df2b.columns))

# ---------- Definitions ----------
shared_pipelines = [
    ('Pipeline_Borda',         'Borda'),
    ('Pipeline_BordaWeighted', 'Borda Weighted'),
    ('Pipeline_CombMAX',       'CombMAX'),
    ('Pipeline_Dedup',         'Dedup'),
]

metrics = [
    ('R@10',    'Recall (R@10)'),
    ('R@c',     'Recall (R@c)'),
    ('nDCG@10', 'nDCG@10'),
]

x_col = 'n'

# =============================================
# 1. Per-metric line plots
# =============================================
cmap = plt.cm.tab10

for metric_suffix, metric_label in metrics:
    fig, ax = plt.subplots(figsize=(12, 7))
    color_idx = 0

    for prefix, label in shared_pipelines:
        col = f'{prefix}_{metric_suffix}'
        if col in df2.columns and col in df2b.columns:
            color = cmap(color_idx)
            ax.plot(df2[x_col],  df2[col],  '-o',  color=color,
                    label=f'{label} (Merge→Rerank)', linewidth=2, markersize=6)
            ax.plot(df2b[x_col], df2b[col], '--s', color=color,
                    label=f'{label} (Rerank→Merge)', linewidth=2, markersize=6)
            color_idx += 1

    # BM25 + MonoT5 baseline reference line
    baseline_col = f'BM25_MonoT5_{metric_suffix}'
    if baseline_col in df2.columns:
        ax.axhline(y=df2[baseline_col].iloc[-1], color='gray', linestyle=':',
                   linewidth=1.5, label='BM25 + MonoT5 Baseline')

    ax.set_xlabel('Number of Rewrites (n)', fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_title(f'Exp 2 vs 2b: Merge→Rerank vs Rerank→Merge ({metric_label})',
                 fontsize=16)
    ax.legend(fontsize=10, ncol=2, loc='best', framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = os.path.join(PLOT_DIR, f'exp2_vs_2b_{metric_suffix}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved: {fname}')
    plt.show()

# =============================================
# 2. Grouped bar charts at n=10
# =============================================
n_val = 10
row2  = df2[df2[x_col]  == n_val].iloc[0]
row2b = df2b[df2b[x_col] == n_val].iloc[0]

for metric_suffix, metric_label in metrics:
    labels_bar = [lbl for _, lbl in shared_pipelines]
    vals_2  = [row2[f'{pfx}_{metric_suffix}']
               for pfx, _ in shared_pipelines if f'{pfx}_{metric_suffix}' in df2.columns]
    vals_2b = [row2b[f'{pfx}_{metric_suffix}']
               for pfx, _ in shared_pipelines if f'{pfx}_{metric_suffix}' in df2b.columns]

    x_bar = np.arange(len(labels_bar))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x_bar - width / 2, vals_2,  width,
                   label='Exp 2: Merge→Rerank', color='#4C72B0', edgecolor='white')
    bars2 = ax.bar(x_bar + width / 2, vals_2b, width,
                   label='Exp 2b: Rerank→Merge', color='#DD8452', edgecolor='white')

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Fusion Method', fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_title(f'Exp 2 vs 2b at n={n_val}: {metric_label}', fontsize=16)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(labels_bar, fontsize=12)
    ax.legend(fontsize=12, framealpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    fname = os.path.join(PLOT_DIR, f'exp2_vs_2b_bar_{metric_suffix}_n{n_val}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved: {fname}')
    plt.show()

# =============================================
# 3. Percentage difference table
# =============================================
print('\n' + '=' * 70)
print(f'  Percentage Difference at n={n_val}  (positive = Exp2b better)')
print('=' * 70)
print(f'{"Method":<22} {"R@10":>12} {"R@c":>12} {"nDCG@10":>12}')
print('-' * 70)
for prefix, label in shared_pipelines:
    diffs = []
    for metric_suffix, _ in metrics:
        col = f'{prefix}_{metric_suffix}'
        v2  = row2[col]  if col in df2.columns  else float('nan')
        v2b = row2b[col] if col in df2b.columns else float('nan')
        pct = ((v2b - v2) / v2) * 100 if v2 != 0 else float('nan')
        diffs.append(f'{pct:+.2f}%')
    print(f'{label:<22} {diffs[0]:>12} {diffs[1]:>12} {diffs[2]:>12}')
print('=' * 70)
