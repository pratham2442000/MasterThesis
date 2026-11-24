"""
Generate Thesis Plots
=====================
Re-plots all experiment results from existing CSVs.
No experiment re-runs needed.

Output structure:
    results_for_thesis/
    ├── exp1/
    │   ├── wiki/
    │   │   ├── R@10.png
    │   │   └── nDCG@10.png
    │   ├── triviaqa/
    │   └── msmarco/
    ├── exp2/
    │   ├── wiki/
    │   ├── triviaqa/
    │   └── msmarco/
    ...

Usage:
    conda activate mt
    python generate_thesis_plots.py              # all experiments
    python generate_thesis_plots.py --exp exp1    # single experiment
    python generate_thesis_plots.py --dpi 600     # custom DPI
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ─── Global Plot Style ──────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
})

# ─── Project Paths ──────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUTPUT = os.path.join(BASE, "results_for_thesis")

# Dataset folder → pretty name mapping
DATASET_MAP = {
    "wiki_musique": "wiki",
    "dpr-w100": "triviaqa",
    "msmarco-passage": "msmarco",
}

# ─── Style Definitions ─────────────────────────────────────

# Use matplotlib's default tab10 color cycle (old style)
# Markers cycle to distinguish overlapping lines
MARKERS = ["o", "x", "s", "^", "D", "v", "P", "X", "*", "p"]


# ─── Pretty Labels ──────────────────────────────────────────

_RERANKER_NAME_MAP = {
    "cross_encoder": "MiniLM-L6",
    "minilm": "MiniLM-L6",
    "monot5": "MonoT5",
}

LABEL_MAP = {
    "BM25": "BM25 Baseline",
    "BM25_MonoT5": "BM25 + MonoT5",
    "BM25_MiniLM-L6": "BM25 + MiniLM-L6",
    "Dense": "Dense (TCT-ColBERT)",
    "Dense_MonoT5": "Dense + MonoT5",
    "Dense_MiniLM-L6": "Dense + MiniLM-L6",
    "Pipeline_DedupCutoff": "Dedup+Cutoff",
    "Pipeline_Borda": "Borda",
    "Pipeline_BordaWeighted": "Borda (Weighted)",
    "Pipeline_CombMAX": "CombMAX",
    "Pipeline_RRF": "RRF",
    "Pipeline_Exp4Fuse": "Exp4Fuse",
    "ORE_Adaptive": "ORE Adaptive",
    "ReformIR": "ReformIR",
    "Oracle": "Oracle (Upper Bound)",
    "DedupOracle": "Dedup Oracle",
    "BestRewrite": "Best Rewrite",
    "Full_RAG_with_reranker": "Full RAG (+ Reranker)",
    "Full_RAG_without_reranker": "Full RAG (No Reranker)",
}

baselines = (
    "BM25", 
    "Dense",
    # "BM25_MonoT5",
    # "BM25_MiniLM-L6",
    # "Dense_MonoT5",
    # "Dense_MiniLM-L6",
    )

def _update_label_map(reranker: Optional[str], exp_name: str = "") -> None:
    """Patch LABEL_MAP with reranker-specific pretty names.
    For exp1, skip MonoT5→MiniLM remap since exp1 CSV has both rerankers.
    """
    if not reranker:
        return
    pretty = _RERANKER_NAME_MAP.get(reranker.lower(), reranker)
    LABEL_MAP[f"BM25_{reranker}"] = f"BM25 + {pretty}"
    LABEL_MAP[f"Dense_{reranker}"] = f"Dense + {pretty}"

    # Also cover MiniLM-L6 style keys if reranker resolves to it
    if pretty != reranker:
        LABEL_MAP[f"BM25_{pretty}"] = f"BM25 + {pretty}"
        LABEL_MAP[f"Dense_{pretty}"] = f"Dense + {pretty}"

    # Exp2/2a/2b CSVs have columns named *_MonoT5 even when run with MiniLM.
    # Remap those legacy keys — but NOT for exp1 which has both rerankers.
    if exp_name != "exp1":
        _MINILM_RERANKERS = {"minilm", "cross_encoder"}
        if reranker.lower() in _MINILM_RERANKERS:
            LABEL_MAP["BM25_MonoT5"] = f"BM25 + {pretty}"
            LABEL_MAP["Dense_MonoT5"] = f"Dense + {pretty}"


def _pretty_label(system: str) -> str:
    return LABEL_MAP.get(system, system.replace("_", " "))


# ─── Metadata Helpers ──────────────────────────────────────

import json
import glob


def _find_metadata(csv_path: str) -> Optional[str]:
    """Find the metadata JSON file next to a CSV.

    Tries (in order):
      1. Same basename with _metadata.json suffix
      2. Any *metadata*.json in the same directory
      3. Any *metadata*.json in the parent directory (for exp4/exp7 mode subdirs)
    """
    csv_dir = os.path.dirname(csv_path)
    csv_base = os.path.splitext(os.path.basename(csv_path))[0]

    # Try exact match: experiment_1_results → experiment_1_results_metadata.json
    exact = os.path.join(csv_dir, csv_base + "_metadata.json")
    if os.path.isfile(exact):
        return exact

    # Try any metadata json in same dir
    candidates = glob.glob(os.path.join(csv_dir, "*metadata*.json"))
    if candidates:
        return candidates[0]

    # Try parent dir (exp4/exp7: CSV is in run_dir/linear/, metadata is in run_dir/)
    parent = os.path.dirname(csv_dir)
    candidates = glob.glob(os.path.join(parent, "*metadata*.json"))
    if candidates:
        return candidates[0]

    return None


def _read_metadata(meta_path: Optional[str]) -> Dict:
    """Read metadata JSON, return dict. Empty dict if not found."""
    if not meta_path or not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _reranker_pretty_from_metadata(meta: Dict) -> Optional[str]:
    """Extract a pretty reranker name from metadata."""
    rtype = meta.get("reranker_type", "")
    rmodel = meta.get("reranker_model_name", "")
    if "monot5" in rmodel.lower() or rtype == "monot5":
        return "MonoT5"
    if "minilm" in rmodel.lower() or rtype == "cross_encoder":
        return "MiniLM-L6"
    if rtype:
        return rtype
    return None


# ─── CSV Registry ──────────────────────────────────────────

def _find_all_csvs(
    dataset_dir: str,
    exp_subdir: str,
    reranker: Optional[str] = None,
) -> List[Dict]:
    """Find ALL CSVs in an experiment directory, grouped by reranker.

    Returns a list of {"csv": path, "metadata": meta_dict} dicts,
    one per unique reranker found. For each reranker, picks the best
    run (prefers 'final' dirs, then latest mtime).
    If --reranker is specified, only returns runs matching that filter.
    """
    exp_path = os.path.join(dataset_dir, exp_subdir)
    if not os.path.isdir(exp_path):
        return []

    # Collect all run dirs with their metadata
    candidates = []  # (is_final, mtime, csv_path, metadata)
    for entry in os.listdir(exp_path):
        run_dir = os.path.join(exp_path, entry)
        if not os.path.isdir(run_dir):
            continue
        if entry in ("old", "OLD", "plots"):
            continue
        for f in os.listdir(run_dir):
            if f.endswith(".csv") and "experiment_" in f and "metadata" not in f:
                csv_path = os.path.join(run_dir, f)
                entry_lower = entry.lower()
                is_final = "final" in entry_lower
                mtime = os.path.getmtime(csv_path)
                meta_path = _find_metadata(csv_path)
                meta = _read_metadata(meta_path)
                candidates.append((is_final, mtime, csv_path, meta))

    if not candidates:
        return []

    # Group by reranker type
    by_reranker = {}  # reranker_key -> list of (is_final, mtime, csv_path, meta)
    for is_final, mtime, csv_path, meta in candidates:
        rr_key = _reranker_pretty_from_metadata(meta) or "unknown"
        by_reranker.setdefault(rr_key, []).append((is_final, mtime, csv_path, meta))

    # If --reranker filter active, filter groups
    if reranker:
        filtered = {}
        for rr_key, runs in by_reranker.items():
            if reranker.lower() in rr_key.lower():
                filtered[rr_key] = runs
        by_reranker = filtered

    # Pick best run per reranker
    results = []
    for rr_key, runs in sorted(by_reranker.items()):
        runs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = runs[0]
        results.append({"csv": best[2], "metadata": best[3]})

    return results


def _find_mode_csvs(
    exp_path: str,
) -> Optional[Dict]:
    """Find CSVs for exp4/exp7 multi-mode experiments.

    Returns {"modes": {ce_mode: csv_path, ...}, "metadata": meta_dict} or None.
    """
    if not os.path.isdir(exp_path):
        return None

    mode_csvs = {}
    meta = {}
    for ce_mode in ("linear", "fixed"):
        csv_found = None
        for entry in sorted(os.listdir(exp_path), reverse=True):
            mode_dir = os.path.join(exp_path, entry, ce_mode)
            if os.path.isdir(mode_dir):
                for f in os.listdir(mode_dir):
                    if f.endswith(".csv") and "metadata" not in f:
                        is_final = "final" in entry.lower()
                        if csv_found is None or is_final:
                            csv_found = os.path.join(mode_dir, f)
                            if is_final:
                                break
            if csv_found and "final" in csv_found.lower():
                break
        if csv_found:
            mode_csvs[ce_mode] = csv_found

    if not mode_csvs:
        return None

    # Find metadata: try parent run dir, then any mode CSV's parent
    any_csv = list(mode_csvs.values())[0]
    meta_path = _find_metadata(any_csv)
    meta = _read_metadata(meta_path)

    return {"modes": mode_csvs, "metadata": meta}


def build_csv_registry(reranker: Optional[str] = None) -> Dict[str, Dict[str, List[Dict]]]:
    """Build registry: {exp_name: {dataset_pretty: [entry_dict, ...]}}

    Each entry_dict is either:
      {"csv": path, "metadata": {...}}           (simple experiments)
      {"modes": {"linear": path, ...}, "metadata": {...}}  (mode experiments)

    Multiple entries per dataset = same experiment run with different rerankers.
    """

    registry = {}

    for dataset_folder, pretty_name in DATASET_MAP.items():
        dataset_dir = os.path.join(RESULTS, dataset_folder)
        if not os.path.isdir(dataset_dir):
            continue

        # ── Simple experiments ──
        for exp_name in ("exp1", "exp2", "exp2a", "exp2b", "exp2_noise", "exp5"):
            results = _find_all_csvs(dataset_dir, exp_name, reranker=reranker)
            for result in results:
                registry.setdefault(exp_name, {}).setdefault(pretty_name, []).append(result)

        # ── Mode experiments (exp4, exp7) ──
        for exp_name in ("exp4", "exp7"):
            exp_path = os.path.join(dataset_dir, exp_name)
            result = _find_mode_csvs(exp_path)
            if result:
                registry.setdefault(exp_name, {}).setdefault(pretty_name, []).append(result)

    return registry


# ─── Plot Definitions Per Experiment ────────────────────────

def _detect_systems(df: pd.DataFrame, metric_suffix: str) -> List[str]:
    """Detect system names from column names ending with metric_suffix."""
    systems = []
    for col in df.columns:
        if col.endswith(metric_suffix):
            system = col[: -len(metric_suffix)]
            if system:
                systems.append(system)
    return systems


def _plot_single(
    df: pd.DataFrame,
    x_col: str,
    systems: List[str],
    metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    use_log_scale: bool = False,
    figsize: Tuple[float, float] = (8, 5.5),
    dpi: int = 300,
    include_legend: bool = True,
):
    """Plot a single metric for multiple systems."""
    fig, ax = plt.subplots(figsize=figsize)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sys_idx = 0

    for system in systems:
        col = f"{system}_{metric}"
        if col not in df.columns:
            continue

        y_vals = pd.to_numeric(df[col], errors="coerce")
        x_vals = df[x_col]

        # Skip if all NaN
        mask = y_vals.notna()
        if mask.sum() == 0:
            continue

        label = _pretty_label(system)

        if system in baselines:
            color = "#888888" # Gray for baselines
            marker = "o" if system == "BM25" else "^"
            linestyle = "-" if system == "BM25" else "--"
        else:
            color = color_cycle[sys_idx % len(color_cycle)]
            marker = MARKERS[sys_idx % len(MARKERS)]
            linestyle = "-"
            sys_idx += 1

        ax.plot(
            x_vals[mask],
            y_vals[mask],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=7,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=15)

    if use_log_scale:
        ax.set_xscale("log")
        # Nice ticks for log scale
        _nice_ticks = [10, 16, 27, 46, 77, 129, 215, 359, 599, 1000]
        x_min, x_max = df[x_col].min(), df[x_col].max()
        ticks = [t for t in _nice_ticks if x_min * 0.8 <= t <= x_max * 1.2]
        if len(ticks) >= 3:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    if include_legend:
        ax.legend(
            loc="best",
            framealpha=0.9,
            edgecolor="#cccccc",
            fancybox=True,
            fontsize=12,
            ncol=2
        )
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _desaturate(hex_color: str, factor: float = 0.25) -> str:
    """Desaturate a hex color toward gray. factor=0 is full gray, 1 is original."""
    import colorsys
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = s * factor
    v = min(1.0, v * 0.85)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))


def _plot_noise(
    df: pd.DataFrame,
    x_col: str,
    systems: List[str],
    metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    figsize: Tuple[float, float] = (8, 5.5),
    dpi: int = 300,
):
    """Noise analysis plot.
    Oracle/DedupOracle/BestRewrite: thick vivid, directly labeled.
    Fusion methods: shaded band (min–max) + dashed mean.
    Baselines: distinct gray lines with markers.
    """
    KEY_SYSTEMS = {"Oracle", "DedupOracle", "BestRewrite"}
    DIRECT_LABEL_MAP = {
        "Oracle": "Oracle",
        "DedupOracle": "Dedup Oracle",
        "BestRewrite": "Best Rewrite",
    }
    BASELINE_SYSTEMS = {"BM25", "Dense"}

    fig, ax = plt.subplots(figsize=figsize)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    key_colors = {"Oracle": color_cycle[0], "DedupOracle": color_cycle[3], "BestRewrite": color_cycle[2]}
    plotted_curves = {}  # system -> (x_end, y_end, color)

    # ── 1. Collect fusion method values for shaded band ──
    fusion_vals = []
    x_vals = df[x_col]
    for system in systems:
        if system in KEY_SYSTEMS or system in BASELINE_SYSTEMS:
            continue
        # Also skip baseline+reranker combos
        if any(system.startswith(bl) and system != bl for bl in BASELINE_SYSTEMS):
            # This is a reranker baseline like BM25_MonoT5 — treat as baseline
            continue
        col = f"{system}_{metric}"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").values
            fusion_vals.append(vals)

    if fusion_vals:
        fusion_arr = np.array(fusion_vals)
        y_min = np.nanmin(fusion_arr, axis=0)
        y_max = np.nanmax(fusion_arr, axis=0)
        y_mean = np.nanmean(fusion_arr, axis=0)
        ax.fill_between(x_vals, y_min, y_max, alpha=0.3, color="#999999",
                        label="Fusion Methods (range)", zorder=1)
        ax.plot(x_vals, y_mean, color="#666666", linewidth=1.5, linestyle="--",
                alpha=0.7, label="Fusion Methods (mean)", zorder=2)

    # ── 2. Baselines (BM25, BM25+reranker, Dense, Dense+reranker) ──
    baseline_styles = {
        "BM25": ("o", "-", "BM25 Baseline"),
        "Dense": ("^", "-", "Dense Baseline"),
    }
    for system in systems:
        if system in BASELINE_SYSTEMS:
            col = f"{system}_{metric}"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            mask = vals.notna()
            if mask.sum() == 0:
                continue
            marker, ls, lbl = baseline_styles.get(system, ("x", "-", system))
            ax.plot(x_vals[mask], vals[mask], color="#555555", marker=marker,
                    linestyle=ls, linewidth=1.8, markersize=6, alpha=0.8,
                    label=lbl, zorder=3)
            plotted_curves[system] = (x_vals[mask].iloc[-1], vals[mask].iloc[-1], "#555555")
        elif any(system.startswith(bl + "_") for bl in BASELINE_SYSTEMS):
            # Reranker baselines like BM25_MonoT5
            col = f"{system}_{metric}"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            mask = vals.notna()
            if mask.sum() == 0:
                continue
            label = _pretty_label(system)
            ax.plot(x_vals[mask], vals[mask], color="#555555", marker="s",
                    linestyle="--", linewidth=1.8, markersize=5, alpha=0.8,
                    label=label, zorder=3)
            plotted_curves[system] = (x_vals[mask].iloc[-1], vals[mask].iloc[-1], "#555555")

    # ── 3. Key curves ──
    key_markers = {"Oracle": "o", "DedupOracle": "D", "BestRewrite": "s"}
    for system in ("Oracle", "DedupOracle", "BestRewrite"):
        col = f"{system}_{metric}"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna()
        if mask.sum() == 0:
            continue
        color = key_colors[system]
        lw = 3.0 if system in ("Oracle", "DedupOracle") else 2.5
        ax.plot(x_vals[mask], vals[mask], color=color, linewidth=lw,
                marker=key_markers[system], markersize=7, label=_pretty_label(system),
                zorder=5)
        plotted_curves[system] = (x_vals[mask].iloc[-1], vals[mask].iloc[-1], color)

    # ── 4. Direct labels on key curves ──
    for system, direct_label in DIRECT_LABEL_MAP.items():
        if system in plotted_curves:
            x_end, y_end, color = plotted_curves[system]
            offset_x = -40 if direct_label == "Oracle" else -50
            ax.annotate(
                direct_label,
                xy=(x_end, y_end),
                xytext=(offset_x, -16),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                color=color,
                va="center",
                zorder=6,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=15)

    # ── Secondary x-axis: Avg Documents (from DedupOracle_avg_docs) ──
    if "DedupOracle_avg_docs" in df.columns:
        avg_docs = pd.to_numeric(df["DedupOracle_avg_docs"], errors="coerce")
        n_vals = df[x_col]
        valid = avg_docs.notna() & n_vals.notna()
        if valid.sum() > 0:
            n_arr = n_vals[valid].values
            docs_arr = avg_docs[valid].values

            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(n_arr)
            dedup_color = key_colors["DedupOracle"]
            ax2.set_xticklabels([str(int(round(d))) for d in docs_arr],
                                fontsize=12, rotation=45, color=dedup_color)
            ax2.set_xlabel("Avg. Documents (Dedup)", fontsize=12, color=dedup_color)
            ax2.tick_params(axis="x", colors=dedup_color)

    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─── Experiment Plot Configurations ────────────────────────

def plot_exp1(csv_path: str, out_dir: str, dpi: int, ds_name: str = "",
             reranker_name: str = ""):
    """Exp1: cutoff sweep — R@10, nDCG@10.
    Exp1 CSV contains BOTH rerankers (MiniLM-L6 & MonoT5) so no reranker tag in title.
    """
    df = pd.read_csv(csv_path)
    systems = _detect_systems(df, "_R@10")
    ds_tag = f" — {ds_name}" if ds_name else ""

    _plot_single(
        df, "c", systems, "R@10",
        title=f"Recall@10 vs. Initial Cutoff Depth{ds_tag}",
        xlabel="Initial Cutoff (c)",
        ylabel="Recall@10",
        output_path=os.path.join(out_dir, f"R@10_{ds_name}.png"),
        use_log_scale=True, dpi=dpi,
        include_legend=False,
    )
    _plot_single(
        df, "c", systems, "nDCG@10",
        title=f"nDCG@10 vs. Initial Cutoff Depth{ds_tag}",
        xlabel="Initial Cutoff (c)",
        ylabel="nDCG@10",
        output_path=os.path.join(out_dir, f"nDCG@10_{ds_name}.png"),
        use_log_scale=True, dpi=dpi,
        include_legend=False,
    )

    # Save standalone legend image
    _save_standalone_legend(df, systems, out_dir, dpi)


def _save_standalone_legend(
    df: pd.DataFrame,
    systems: List[str],
    out_dir: str,
    dpi: int = 300,
):
    """Save a standalone horizontal legend image matching _plot_single styles."""
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(1, 1))  # tiny dummy axes

    sys_idx = 0
    handles = []
    labels = []
    for system in systems:
        # Check system actually has data
        if not any(f"{system}_" in c for c in df.columns):
            continue

        label = _pretty_label(system)

        if system in baselines:
            color = "#888888"
            marker = "o" if system == "BM25" else "^"
            linestyle = "-" if system == "BM25" else "--"
        else:
            color = color_cycle[sys_idx % len(color_cycle)]
            marker = MARKERS[sys_idx % len(MARKERS)]
            linestyle = "-"
            sys_idx += 1

        h, = ax.plot([], [], label=label, color=color, marker=marker,
                     linestyle=linestyle, linewidth=2.0, markersize=7)
        handles.append(h)
        labels.append(label)

    ax.set_visible(False)  # hide dummy axes

    legend = fig.legend(
        handles, labels,
        loc="center",
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
        fontsize=14,
        ncol=len(handles),  # single horizontal row
        handlelength=2.5,
        columnspacing=1.5,
    )

    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Add small padding
    bbox = bbox.expanded(1.1, 1.3)

    legend_path = os.path.join(out_dir, "legend.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(legend_path, dpi=dpi, bbox_inches=bbox)
    plt.close(fig)
    print(f"  Saved legend: {legend_path}")


def plot_exp2(csv_path: str, out_dir: str, dpi: int, ds_name: str = "",
             reranker_name: str = ""):
    """Exp2: rewrite sweep — R@10, R@50, nDCG@10."""
    df = pd.read_csv(csv_path)
    systems = _detect_systems(df, "_R@10")
    rr_tag = f" [{reranker_name}]" if reranker_name else ""
    ds_tag = f" — {ds_name}{rr_tag}"

    for metric, ylabel in [("R@10", "Recall@10"), ("R@50", "Recall@50"), ("nDCG@10", "nDCG@10")]:
        _plot_single(
            df, "n", systems, metric,
            title=f"{ylabel} vs. Number of Rewrites{ds_tag}",
            xlabel="Number of Rewrites (n)",
            ylabel=ylabel,
            output_path=os.path.join(out_dir, f"{metric}_{ds_name}.png"),
            use_log_scale=False, dpi=dpi,
            include_legend=False,
        )

    # Save standalone legend image (once per dataset/reranker combo)
    _save_standalone_legend(df, systems, out_dir, dpi)


def plot_exp2a(csv_path: str, out_dir: str, dpi: int, ds_name: str = "",
              reranker_name: str = ""):
    """Exp2a: dense rewrite sweep — same as exp2."""
    plot_exp2(csv_path, out_dir, dpi, ds_name, reranker_name=reranker_name)


def plot_exp2b(csv_path: str, out_dir: str, dpi: int, ds_name: str = "",
              reranker_name: str = ""):
    """Exp2b: BM25 rewrite sweep with MiniLM-L6 — same as exp2."""
    plot_exp2(csv_path, out_dir, dpi, ds_name, reranker_name=reranker_name)


def plot_exp2_noise(csv_path: str, out_dir: str, dpi: int, ds_name: str = "",
                   reranker_name: str = ""):
    """Exp2 noise: oracles vivid, methods desaturated, DedupOracle annotated."""
    df = pd.read_csv(csv_path)
    systems = _detect_systems(df, "_R@10")
    rr_tag = f" [{reranker_name}]" if reranker_name else ""
    ds_tag = f" — {ds_name}{rr_tag}"

    for metric, ylabel in [("R@10", "Recall@10"), ("R@50", "Recall@50"), ("nDCG@10", "nDCG@10")]:
        _plot_noise(
            df, "n", systems, metric,
            title=f"{ylabel} vs. Noise Analysis{ds_tag}",
            xlabel="Number of Rewrites (n)",
            ylabel=ylabel,
            output_path=os.path.join(out_dir, f"{metric}_{ds_name}.png"),
            dpi=dpi,
        )


def _plot_combined_modes(
    mode_csvs: Dict[str, str],
    out_dir: str,
    dpi: int,
    exp_label: str,
    ds_name: str = "",
):
    """Plot both CE modes (linear + fixed) on one graph per metric.

    Systems from each mode get a [L] or [F] suffix in the legend.
    Baselines (BM25, Dense, etc.) are only plotted once (from linear).
    """
    BASELINE_PREFIXES = ("BM25", "Dense")
    MODE_TAG = {"linear": "[L]", "fixed": "[F]"}
    MODE_LINESTYLE = {"linear": "-", "fixed": "--"}

    ds_tag = f" — {ds_name}" if ds_name else ""

    for metric, ylabel in [("R@10", "Recall@10"), ("R@50", "Recall@50"), ("nDCG@10", "nDCG@10")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        sys_color_map = {}  # system base name → color
        color_idx = 0
        baselines_plotted = set()

        for ce_mode in ("linear", "fixed"):
            csv_path = mode_csvs.get(ce_mode)
            if csv_path is None:
                continue
            df = pd.read_csv(csv_path)
            # For R@50, drop rows where c <= 50 (meaningless)
            if metric == "R@50":
                df = df[df["c"] > 50].reset_index(drop=True)
            systems = _detect_systems(df, f"_{metric}")

            for system in systems:
                col = f"{system}_{metric}"
                if col not in df.columns:
                    continue
                y_vals = pd.to_numeric(df[col], errors="coerce")
                x_vals = df["c"]
                mask = y_vals.notna()
                if mask.sum() == 0:
                    continue

                is_baseline = system.startswith(BASELINE_PREFIXES)

                if is_baseline:
                    # Only plot baselines once (from first mode that has them)
                    if system in baselines_plotted:
                        continue
                    baselines_plotted.add(system)
                    label = _pretty_label(system)
                    color = "#888888"
                    # Pure BM25 = dotted, BM25+reranker = dashed
                    if system == "BM25":
                        marker = "o"
                        linestyle = "-"
                    elif "_" in system:  # BM25_MiniLM-L6 etc.
                        marker = "s"
                        linestyle = "-"
                    else:
                        marker = "^"
                        linestyle = "-"
                else:
                    tag = MODE_TAG.get(ce_mode, ce_mode)
                    label = f"{_pretty_label(system)} {tag}"
                    linestyle = MODE_LINESTYLE.get(ce_mode, "-")
                    # Same color for same system across modes
                    if system not in sys_color_map:
                        sys_color_map[system] = color_cycle[color_idx % len(color_cycle)]
                        color_idx += 1
                    color = sys_color_map[system]
                    marker = "o" if ce_mode == "linear" else "x"

                ax.plot(
                    x_vals[mask], y_vals[mask],
                    label=label, color=color, marker=marker,
                    linestyle=linestyle, linewidth=2.0, markersize=7,
                )

        ax.set_xlabel("Initial Cutoff (c)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. Initial Cutoff — {exp_label}{ds_tag}", fontweight="bold", fontsize=15)
        ax.set_xscale("log")
        _nice_ticks = [10, 16, 27, 46, 77, 129, 215, 359, 599, 1000]
        # Use x range from any available CSV
        any_df = pd.read_csv(list(mode_csvs.values())[0])
        x_min, x_max = any_df["c"].min(), any_df["c"].max()
        ticks = [t for t in _nice_ticks if x_min * 0.8 <= t <= x_max * 1.2]
        if len(ticks) >= 3:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.legend(loc="best", framealpha=0.9, edgecolor="#cccccc", fancybox=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{metric}_{ds_name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_exp4(mode_csvs: Dict[str, str], out_dir: str, dpi: int, ds_name: str = "",
             reranker_name: str = ""):
    """Exp4: ORE Adaptive — both CE modes on one graph."""
    _plot_combined_modes(mode_csvs, out_dir, dpi, "ORE", ds_name)


def plot_exp7(mode_csvs: Dict[str, str], out_dir: str, dpi: int, ds_name: str = "",
             reranker_name: str = ""):
    """Exp7: ReformIR — both CE modes on one graph."""
    _plot_combined_modes(mode_csvs, out_dir, dpi, "ReformIR", ds_name)


def plot_exp5(
    csv_path: str,
    out_dir: str,
    dpi: int,
    ds_name: str = "",
    reranker_name: str = "",
):
    """Exp5: RAG evaluation — EM, F1, and LLM-as-Judge vs cutoff."""
    df = pd.read_csv(csv_path)
    rr_tag = f" [{reranker_name}]" if reranker_name else ""
    ds_tag = f" — {ds_name}{rr_tag}"

    # Detect systems from EM columns (simple suffix)
    systems = _detect_systems(df, "_EM")

    # ── Standard metrics: EM, F1 ──
    for metric, ylabel in [("EM", "Exact Match"), ("F1", "F1 Score")]:
        _plot_single(
            df, "c", systems, metric,
            title=f"{ylabel} vs. Initial Cutoff{ds_tag}",
            xlabel="Initial Cutoff (c)",
            ylabel=ylabel,
            output_path=os.path.join(out_dir, f"{metric}_{ds_name}.png"),
            use_log_scale=True, dpi=dpi,
        )

    # ── LLM-as-Judge metric ──
    # Column names contain parens/commas, e.g.:
    #   Full_RAG_with_reranker_LLMasJudge(./models/Llama-3.2-3B-Instruct,agg=max)
    # Find all such columns and plot them.
    llm_cols = [c for c in df.columns if "LLMasJudge" in c]
    if llm_cols:
        # Extract the suffix shared across systems (everything from "LLMasJudge...")
        llm_suffix = None
        for col in llm_cols:
            idx = col.index("LLMasJudge")
            candidate = col[idx:]
            if llm_suffix is None:
                llm_suffix = candidate
            # all should share same suffix

        if llm_suffix:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            for i, col in enumerate(llm_cols):
                # Extract system name: everything before _LLMasJudge
                system = col[:col.index("_LLMasJudge")]
                label = _pretty_label(system)
                y_vals = pd.to_numeric(df[col], errors="coerce")
                x_vals = df["c"]
                mask = y_vals.notna()
                if mask.sum() == 0:
                    continue

                ax.plot(
                    x_vals[mask], y_vals[mask],
                    label=label,
                    color=color_cycle[i % len(color_cycle)],
                    marker=MARKERS[i % len(MARKERS)],
                    linewidth=2.0, markersize=7,
                )

            ax.set_xlabel("Initial Cutoff (c)")
            ax.set_ylabel("LLM-as-Judge Score")
            ax.set_title(f"LLM-as-Judge vs. Initial Cutoff{ds_tag}",
                         fontweight="bold", fontsize=15)
            ax.set_xscale("log")
            _nice_ticks = [10, 16, 27, 46, 77, 129, 215, 359, 599, 1000]
            x_min, x_max = df["c"].min(), df["c"].max()
            ticks = [t for t in _nice_ticks if x_min * 0.8 <= t <= x_max * 1.2]
            if len(ticks) >= 3:
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks])
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.legend(loc="best", framealpha=0.9, edgecolor="#cccccc",
                      fancybox=True, fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            out_path = os.path.join(out_dir, f"LLMasJudge_{ds_name}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ─── Cross-Experiment Comparisons ───────────────────────────

# Shared fusion methods between exp2 (sparse) and exp2a (dense)
_CROSS_SHARED_METHODS = ["Pipeline_Borda", "Pipeline_CombMAX", "Pipeline_RRF", "Pipeline_Exp4Fuse"]

_CROSS_METHOD_LABELS = {
    "Pipeline_Borda": "Borda",
    "Pipeline_CombMAX": "CombMAX",
    "Pipeline_RRF": "RRF",
    "Pipeline_Exp4Fuse": "Exp4Fuse",
}


def _save_cross_legend(
    method_colors: Dict[str, str],
    out_dir: str,
    dpi: int = 300,
):
    """Save a standalone legend image for cross sparse-vs-dense plots."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_visible(False)

    handles = []
    labels = []

    # Row 1: BM25 (sparse) methods
    for method in _CROSS_SHARED_METHODS:
        color = method_colors[method]
        lbl = f"{_CROSS_METHOD_LABELS[method]} (BM25)"
        h = Line2D([], [], color=color, linestyle="--", marker="^",
                   markersize=6, linewidth=2.0, alpha=0.85)
        handles.append(h)
        labels.append(lbl)

    # Row 2: Dense methods
    for method in _CROSS_SHARED_METHODS:
        color = method_colors[method]
        lbl = f"{_CROSS_METHOD_LABELS[method]} (Dense)"
        h = Line2D([], [], color=color, linestyle="-", marker="o",
                   markersize=7, linewidth=2.5)
        handles.append(h)
        labels.append(lbl)

    n_methods = len(_CROSS_SHARED_METHODS)
    legend = fig.legend(
        handles, labels,
        loc="center",
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
        fontsize=14,
        ncol=n_methods,  # one row per retriever type
        handlelength=2.5,
        columnspacing=1.5,
    )

    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1.1, 1.3)

    legend_path = os.path.join(out_dir, "legend.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(legend_path, dpi=dpi, bbox_inches=bbox)
    plt.close(fig)
    print(f"  Saved legend: {legend_path}")


def plot_cross_sparse_vs_dense(
    registry: Dict,
    output_root: str,
    dpi: int = 300,
):
    """Cross-experiment comparison: exp2 (sparse) vs exp2a (dense).

    1) Relative-gain line plots (% Δ from each retriever's own baseline).
    2) Bar chart: absolute metric at peak-n vs BM25+reranker baseline.
    """
    # Collect CSV paths per dataset for exp2 and exp2a (MiniLM-L6 only)
    exp2_by_ds = {}   # dataset_pretty -> csv_path
    exp2a_by_ds = {}

    for ds_pretty, entries in registry.get("exp2", {}).items():
        for entry in entries:
            meta = entry.get("metadata", {})
            rr = _reranker_pretty_from_metadata(meta) or ""
            if "minilm" in rr.lower() or "l6" in rr.lower():
                exp2_by_ds[ds_pretty] = entry["csv"]
                break

    for ds_pretty, entries in registry.get("exp2a", {}).items():
        for entry in entries:
            meta = entry.get("metadata", {})
            rr = _reranker_pretty_from_metadata(meta) or ""
            if "minilm" in rr.lower() or "l6" in rr.lower():
                exp2a_by_ds[ds_pretty] = entry["csv"]
                break

    # Only datasets that appear in BOTH experiments
    shared_datasets = sorted(set(exp2_by_ds) & set(exp2a_by_ds))
    if not shared_datasets:
        print("  ⚠ No shared datasets between exp2 and exp2a — skipping cross plot")
        return

    print(f"  Shared datasets: {shared_datasets}")

    out_dir = os.path.join(output_root, "cross_sparse_vs_dense")
    os.makedirs(out_dir, exist_ok=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {m: color_cycle[i] for i, m in enumerate(_CROSS_SHARED_METHODS)}

    DS_TITLES = {"wiki": "Wiki-MuSiQue", "msmarco": "MS MARCO", "triviaqa": "TriviaQA"}

    # ── 1. Relative-gain line plots ──
    for metric, ylabel_abs, ylabel_rel in [
        ("R@50", "Recall@50", "Relative Δ R@50 (%)"),
        ("nDCG@10", "nDCG@10", "Relative Δ nDCG@10 (%)"),
        ("R@10", "Recall@10", "Relative Δ R@10 (%)"),
    ]:
        n_ds = len(shared_datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5.5), squeeze=False)

        for col_idx, ds in enumerate(shared_datasets):
            ax = axes[0, col_idx]

            df_sparse = pd.read_csv(exp2_by_ds[ds])
            df_dense = pd.read_csv(exp2a_by_ds[ds])

            sparse_baseline_col = f"BM25_MonoT5_{metric}"
            dense_baseline_col = f"Dense_MonoT5_{metric}"

            sparse_base = pd.to_numeric(df_sparse[sparse_baseline_col], errors="coerce").iloc[0]
            dense_base = pd.to_numeric(df_dense[dense_baseline_col], errors="coerce").iloc[0]

            ax.axhline(y=0, color="#aaaaaa", linewidth=1.0, linestyle="-", zorder=1)

            for method in _CROSS_SHARED_METHODS:
                color = method_colors[method]
                label_base = _CROSS_METHOD_LABELS[method]

                # Sparse (BM25)
                sparse_col = f"{method}_{metric}"
                if sparse_col in df_sparse.columns:
                    y_abs = pd.to_numeric(df_sparse[sparse_col], errors="coerce")
                    y_rel = ((y_abs - sparse_base) / sparse_base) * 100
                    x = df_sparse["n"]
                    mask = y_rel.notna()
                    ax.plot(
                        x[mask], y_rel[mask],
                        label=f"{label_base}",
                        color=color, linestyle="--", marker="^",
                        linewidth=2.0, markersize=6, alpha=0.85,
                    )

                # Dense (TCT-ColBERT)
                dense_col = f"{method}_{metric}"
                if dense_col in df_dense.columns:
                    y_abs = pd.to_numeric(df_dense[dense_col], errors="coerce")
                    y_rel = ((y_abs - dense_base) / dense_base) * 100
                    x = df_dense["n"]
                    mask = y_rel.notna()
                    ax.plot(
                        x[mask], y_rel[mask],
                        label=f"{label_base} ",  # trailing space distinguishes from sparse
                        color=color, linestyle="-", marker="o",
                        linewidth=2.5, markersize=7,
                    )

            ds_title = DS_TITLES.get(ds, ds)
            ax.set_title(ds_title, fontweight="bold", fontsize=15)
            ax.set_xlabel("Number of Rewrites (n)")
            if col_idx == 0:
                ax.set_ylabel(ylabel_rel)
            ax.grid(True, linestyle="--", alpha=0.3)

        fig.suptitle(
            f"Sparse vs Dense: Relative {ylabel_abs} Gain from Query Rewrite Fusion",
            fontweight="bold", fontsize=16, y=1.02,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"sparse_vs_dense_{metric}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── Standalone legend image ──
    _save_cross_legend(method_colors, out_dir, dpi)


    # ── 2. Bar chart: absolute performance at best-n vs BM25+reranker baseline ──
    _plot_cross_bar_chart(exp2_by_ds, exp2a_by_ds, shared_datasets, out_dir, dpi, DS_TITLES)


def _plot_cross_bar_chart(
    exp2_by_ds: Dict[str, str],
    exp2a_by_ds: Dict[str, str],
    shared_datasets: List[str],
    out_dir: str,
    dpi: int,
    ds_titles: Dict[str, str],
):
    """Bar chart: absolute R@50 at peak rewrite depth, with BM25+reranker baseline."""
    from matplotlib.lines import Line2D
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {m: color_cycle[i] for i, m in enumerate(_CROSS_SHARED_METHODS)}

    for metric, ylabel in [
        ("R@50", "Recall@50"),
        ("R@10", "Recall@10"),
        ("nDCG@10", "nDCG@10"),
    ]:
        n_ds = len(shared_datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(8 * n_ds, 6.5), squeeze=False)

        for col_idx, ds in enumerate(shared_datasets):
            ax = axes[0, col_idx]
            df_sparse = pd.read_csv(exp2_by_ds[ds])
            df_dense = pd.read_csv(exp2a_by_ds[ds])

            # BM25 + reranker baseline (constant across n)
            bm25_rr_col = f"BM25_MonoT5_{metric}"
            bm25_rr_val = pd.to_numeric(df_sparse[bm25_rr_col], errors="coerce").iloc[0]

            # Dense + reranker baseline
            dense_rr_col = f"Dense_MonoT5_{metric}"
            dense_rr_val = pd.to_numeric(df_dense[dense_rr_col], errors="coerce").iloc[0]

            # For each method, pick peak value across all n
            methods = _CROSS_SHARED_METHODS
            n_methods = len(methods)
            bar_width = 0.35
            x_pos = np.arange(n_methods)

            sparse_vals = []
            dense_vals = []
            sparse_peak_n = []
            dense_peak_n = []

            for method in methods:
                # Sparse peak
                scol = f"{method}_{metric}"
                if scol in df_sparse.columns:
                    s_series = pd.to_numeric(df_sparse[scol], errors="coerce")
                    peak_idx = s_series.idxmax()
                    sparse_vals.append(s_series.iloc[peak_idx])
                    sparse_peak_n.append(int(df_sparse["n"].iloc[peak_idx]))
                else:
                    sparse_vals.append(np.nan)
                    sparse_peak_n.append(0)

                # Dense peak
                dcol = f"{method}_{metric}"
                if dcol in df_dense.columns:
                    d_series = pd.to_numeric(df_dense[dcol], errors="coerce")
                    peak_idx = d_series.idxmax()
                    dense_vals.append(d_series.iloc[peak_idx])
                    dense_peak_n.append(int(df_dense["n"].iloc[peak_idx]))
                else:
                    dense_vals.append(np.nan)
                    dense_peak_n.append(0)

            # Plot bars
            bars_sparse = ax.bar(
                x_pos - bar_width / 2, sparse_vals, bar_width,
                label="BM25 (sparse)", color="#7faed4", edgecolor="#3a6b8c",
                linewidth=0.8, zorder=3,
            )
            bars_dense = ax.bar(
                x_pos + bar_width / 2, dense_vals, bar_width,
                label="Dense (TCT-ColBERT)", color="#e8915a", edgecolor="#b05c2a",
                linewidth=0.8, zorder=3,
            )

            # Value labels on bars (value only, peak-n in caption)
            for bar, val, pn in zip(bars_sparse, sparse_vals, sparse_peak_n):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"{val:.3f}",
                            ha="center", va="bottom", fontsize=13, color="#3a6b8c")

            for bar, val, pn in zip(bars_dense, dense_vals, dense_peak_n):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"{val:.3f}",
                            ha="center", va="bottom", fontsize=13, color="#b05c2a")

            # Baseline lines
            ax.axhline(y=bm25_rr_val, color="#3a6b8c", linewidth=1.8,
                        linestyle="--", zorder=2, alpha=0.8)
            ax.axhline(y=dense_rr_val, color="#b05c2a", linewidth=1.8,
                        linestyle="--", zorder=2, alpha=0.8)

            # Baseline labels on right edge
            ax.text(n_methods - 0.5, bm25_rr_val, f" BM25+RR ({bm25_rr_val:.3f})",
                    va="bottom", ha="left", fontsize=12, color="#3a6b8c", fontweight="bold")
            ax.text(n_methods - 0.5, dense_rr_val, f" Dense+RR ({dense_rr_val:.3f})",
                    va="top", ha="left", fontsize=12, color="#b05c2a", fontweight="bold")

            # Add headroom so annotations don't hit title
            all_vals = [v for v in sparse_vals + dense_vals if not np.isnan(v)]
            if all_vals:
                ax.set_ylim(top=max(all_vals) * 1.10)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([_CROSS_METHOD_LABELS[m] for m in methods], fontsize=14, rotation=0, ha="center")
            ax.set_title(ds_titles.get(ds, ds), fontweight="bold", fontsize=18)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")

            # Legend
            ax.legend(loc="lower left", framealpha=0.9, edgecolor="#cccccc",
                      fancybox=True, fontsize=13)

        fig.suptitle(
            f"Peak Fusion {ylabel} vs Baselines — Sparse vs Dense",
            fontweight="bold", fontsize=18, y=1.02,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"bar_sparse_vs_dense_{metric}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─── Cross-Experiment: MTR vs RTM (exp2 vs exp2b) ──────────

# All 6 fusion methods shared between exp2 and exp2b
_RTM_MTR_METHODS = [
    "Pipeline_Borda", "Pipeline_CombMAX", "Pipeline_RRF",
    "Pipeline_Exp4Fuse", "Pipeline_DedupCutoff", "Pipeline_BordaWeighted",
]

_RTM_MTR_METHOD_LABELS = {
    "Pipeline_Borda": "Borda",
    "Pipeline_CombMAX": "CombMAX",
    "Pipeline_RRF": "RRF",
    "Pipeline_Exp4Fuse": "Exp4Fuse",
    "Pipeline_DedupCutoff": "DedupCutoff",
    "Pipeline_BordaWeighted": "BordaWeighted",
}


def _save_rtm_mtr_legend(method_colors: Dict[str, str], out_dir: str, dpi: int = 300):
    """Standalone legend for RTM vs MTR plots."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_visible(False)

    handles, labels = [], []

    # Row 1: MTR / exp2 (solid)
    for method in _RTM_MTR_METHODS:
        color = method_colors[method]
        lbl = f"{_RTM_MTR_METHOD_LABELS[method]} (MTR)"
        handles.append(Line2D([], [], color=color, linestyle="-", marker="o",
                              markersize=7, linewidth=2.5))
        labels.append(lbl)

    # Row 2: RTM / exp2b (dashed)
    for method in _RTM_MTR_METHODS:
        color = method_colors[method]
        lbl = f"{_RTM_MTR_METHOD_LABELS[method]} (RTM)"
        handles.append(Line2D([], [], color=color, linestyle="--", marker="^",
                              markersize=6, linewidth=2.0, alpha=0.85))
        labels.append(lbl)

    n_methods = len(_RTM_MTR_METHODS)
    legend = fig.legend(
        handles, labels, loc="center",
        framealpha=0.95, edgecolor="#cccccc", fancybox=True,
        fontsize=14, ncol=n_methods, handlelength=2.5, columnspacing=1.5,
    )

    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox = bbox.expanded(1.1, 1.3)

    legend_path = os.path.join(out_dir, "legend.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(legend_path, dpi=dpi, bbox_inches=bbox)
    plt.close(fig)
    print(f"  Saved legend: {legend_path}")


def _plot_rtm_mtr_bar_chart(
    mtr_by_ds: Dict[str, str],
    rtm_by_ds: Dict[str, str],
    shared_datasets: List[str],
    out_dir: str,
    dpi: int,
    ds_titles: Dict[str, str],
):
    """Bar chart: absolute metrics at peak-n for MTR vs RTM, with BM25+reranker baseline.

    Uses only 4 core methods (Borda, CombMAX, RRF, DedupCutoff) to avoid
    overcrowding. BordaWeighted ≈ Borda and Exp4Fuse ≈ RRF per thesis text.
    """
    # Reduced set: drop BordaWeighted and Exp4Fuse (redundant per thesis analysis)
    _BAR_METHODS = [
        "Pipeline_Borda", "Pipeline_CombMAX", "Pipeline_RRF", "Pipeline_DedupCutoff",
    ]
    _BAR_LABELS = {
        "Pipeline_Borda": "Borda",
        "Pipeline_CombMAX": "CombMAX",
        "Pipeline_RRF": "RRF",
        "Pipeline_DedupCutoff": "DedupCutoff",
    }

    for metric, ylabel in [("R@50", "Recall@50"), ("nDCG@10", "nDCG@10"), ("R@10", "Recall@10")]:
        n_ds = len(shared_datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(8 * n_ds, 6.5), squeeze=False)

        for col_idx, ds in enumerate(shared_datasets):
            ax = axes[0, col_idx]
            df_mtr = pd.read_csv(mtr_by_ds[ds])
            df_rtm = pd.read_csv(rtm_by_ds[ds])

            # Shared baseline (BM25 + reranker at n=1)
            baseline_col = f"BM25_MonoT5_{metric}"
            baseline_val = pd.to_numeric(df_mtr[baseline_col], errors="coerce").iloc[0]

            n_methods = len(_BAR_METHODS)
            bar_width = 0.35
            x_pos = np.arange(n_methods)

            mtr_vals, rtm_vals = [], []
            mtr_peak_n, rtm_peak_n = [], []

            for method in _BAR_METHODS:
                col = f"{method}_{metric}"
                # MTR peak
                if col in df_mtr.columns:
                    s = pd.to_numeric(df_mtr[col], errors="coerce")
                    pi = s.idxmax()
                    mtr_vals.append(s.iloc[pi])
                    mtr_peak_n.append(int(df_mtr["n"].iloc[pi]))
                else:
                    mtr_vals.append(np.nan); mtr_peak_n.append(0)
                # RTM peak
                if col in df_rtm.columns:
                    s = pd.to_numeric(df_rtm[col], errors="coerce")
                    pi = s.idxmax()
                    rtm_vals.append(s.iloc[pi])
                    rtm_peak_n.append(int(df_rtm["n"].iloc[pi]))
                else:
                    rtm_vals.append(np.nan); rtm_peak_n.append(0)

            bars_mtr = ax.bar(
                x_pos - bar_width / 2, mtr_vals, bar_width,
                label="MTR", color="#5b9bd5", edgecolor="#2e75b6",
                linewidth=0.8, zorder=3,
            )
            bars_rtm = ax.bar(
                x_pos + bar_width / 2, rtm_vals, bar_width,
                label="RTM", color="#ed7d31", edgecolor="#c55a11",
                linewidth=0.8, zorder=3,
            )

            # Annotations: value only (peak-n described in caption)
            for bar, val, pn in zip(bars_mtr, mtr_vals, mtr_peak_n):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=13, color="#2e75b6")
            for bar, val, pn in zip(bars_rtm, rtm_vals, rtm_peak_n):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=13, color="#c55a11")

            # Baseline line
            ax.axhline(y=baseline_val, color="#666666", linewidth=1.8,
                       linestyle="--", zorder=2, alpha=0.8)
            ax.text(n_methods - 0.4, baseline_val, f"BM25+RR ({baseline_val:.3f})",
                    va="center", ha="center", fontsize=15, color="#666666", fontweight="bold", rotation=-90)

            # Add headroom so annotations don't hit title
            y_max = max(
                max(v for v in mtr_vals if not np.isnan(v)),
                max(v for v in rtm_vals if not np.isnan(v)),
            )
            ax.set_ylim(top=y_max * 1.12)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([_BAR_LABELS[m] for m in _BAR_METHODS],
                               fontsize=14, rotation=30, ha="right")
            ax.set_title(ds_titles.get(ds, ds), fontweight="bold", fontsize=18)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            ax.legend(loc="lower left", framealpha=0.9, edgecolor="#cccccc",
                      fancybox=True, fontsize=13)

        fig.suptitle(
            f"Peak Fusion {ylabel} vs Baseline — MTR vs RTM",
            fontweight="bold", fontsize=18, y=1.02,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"bar_rtm_vs_mtr_{metric}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_cross_rtm_vs_mtr(
    registry: Dict,
    output_root: str,
    dpi: int = 300,
):
    """Cross-experiment: exp2 (MTR) vs exp2b (RTM).

    MTR = Merge-Then-Rerank (exp2): fuse BM25 results first, then rerank.
    RTM = Rerank-Then-Merge (exp2b): rerank per rewrite, then fuse.
    Both use BM25 + cross-encoder reranker. MiniLM-L6 only.
    """
    mtr_by_ds = {}
    rtm_by_ds = {}

    for ds_pretty, entries in registry.get("exp2", {}).items():
        for entry in entries:
            meta = entry.get("metadata", {})
            rr = _reranker_pretty_from_metadata(meta) or ""
            if "minilm" in rr.lower() or "l6" in rr.lower():
                mtr_by_ds[ds_pretty] = entry["csv"]
                break

    for ds_pretty, entries in registry.get("exp2b", {}).items():
        for entry in entries:
            meta = entry.get("metadata", {})
            rr = _reranker_pretty_from_metadata(meta) or ""
            if "minilm" in rr.lower() or "l6" in rr.lower():
                rtm_by_ds[ds_pretty] = entry["csv"]
                break

    shared_datasets = sorted(set(mtr_by_ds) & set(rtm_by_ds))
    if not shared_datasets:
        print("  ⚠ No shared datasets between exp2 and exp2b — skipping")
        return

    print(f"  Shared datasets: {shared_datasets}")

    out_dir = os.path.join(output_root, "cross_rtm_vs_mtr")
    os.makedirs(out_dir, exist_ok=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(_RTM_MTR_METHODS)}

    DS_TITLES = {"wiki": "Wiki-MuSiQue", "msmarco": "MS MARCO", "triviaqa": "TriviaQA"}

    # ── 1. Relative-gain line plots ──
    for metric, ylabel_abs, ylabel_rel in [
        ("R@50", "Recall@50", "Relative Δ R@50 (%)"),
        ("nDCG@10", "nDCG@10", "Relative Δ nDCG@10 (%)"),
        ("R@10", "Recall@10", "Relative Δ R@10 (%)"),
    ]:
        n_ds = len(shared_datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5.5), squeeze=False)

        for col_idx, ds in enumerate(shared_datasets):
            ax = axes[0, col_idx]

            df_mtr = pd.read_csv(mtr_by_ds[ds])
            df_rtm = pd.read_csv(rtm_by_ds[ds])

            # Shared baseline: BM25 + reranker at n=1
            baseline_col = f"BM25_MonoT5_{metric}"
            baseline = pd.to_numeric(df_mtr[baseline_col], errors="coerce").iloc[0]

            ax.axhline(y=0, color="#aaaaaa", linewidth=1.0, linestyle="-", zorder=1)

            for method in _RTM_MTR_METHODS:
                color = method_colors[method]
                label_base = _RTM_MTR_METHOD_LABELS[method]
                col = f"{method}_{metric}"

                # MTR (exp2) — solid
                if col in df_mtr.columns:
                    y_abs = pd.to_numeric(df_mtr[col], errors="coerce")
                    y_rel = ((y_abs - baseline) / baseline) * 100
                    x = df_mtr["n"]
                    mask = y_rel.notna()
                    ax.plot(x[mask], y_rel[mask],
                            color=color, linestyle="-", marker="o",
                            linewidth=2.5, markersize=7)

                # RTM (exp2b) — dashed
                if col in df_rtm.columns:
                    y_abs = pd.to_numeric(df_rtm[col], errors="coerce")
                    y_rel = ((y_abs - baseline) / baseline) * 100
                    x = df_rtm["n"]
                    mask = y_rel.notna()
                    ax.plot(x[mask], y_rel[mask],
                            color=color, linestyle="--", marker="^",
                            linewidth=2.0, markersize=6, alpha=0.85)

            ds_title = DS_TITLES.get(ds, ds)
            ax.set_title(ds_title, fontweight="bold", fontsize=15)
            ax.set_xlabel("Number of Rewrites (n)")
            if col_idx == 0:
                ax.set_ylabel(ylabel_rel)
            ax.grid(True, linestyle="--", alpha=0.3)

        fig.suptitle(
            f"MTR vs RTM: Relative {ylabel_abs} Gain from Query Rewrite Fusion",
            fontweight="bold", fontsize=16, y=1.02,
        )
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"rtm_vs_mtr_{metric}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── 2. Standalone legend ──
    _save_rtm_mtr_legend(method_colors, out_dir, dpi)

    # ── 3. Bar chart ──
    _plot_rtm_mtr_bar_chart(mtr_by_ds, rtm_by_ds, shared_datasets, out_dir, dpi, DS_TITLES)


# ─── Dispatcher ─────────────────────────────────────────────

EXPERIMENT_PLOTTERS = {
    "exp1": plot_exp1,
    "exp2": plot_exp2,
    "exp2a": plot_exp2a,
    "exp2b": plot_exp2b,
    "exp2_noise": plot_exp2_noise,
    "exp4": plot_exp4,
    "exp5": plot_exp5,
    "exp7": plot_exp7,
}

# Cross-experiment comparisons (need full registry, not single CSV)
CROSS_EXPERIMENT_PLOTTERS = {
    "cross_sparse_dense": plot_cross_sparse_vs_dense,
    "cross_rtm_mtr": plot_cross_rtm_vs_mtr,
}


def main():
    parser = argparse.ArgumentParser(description="Generate thesis-quality plots from CSVs")
    parser.add_argument("--exp", type=str, default=None,
                        help="Single experiment to plot (e.g. exp1, exp2, exp4_linear)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (default: 300)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results_for_thesis)")
    parser.add_argument("--reranker", type=str, default=None,
                        help="Filter 'final' run dirs by reranker name substring "
                             "(e.g. monot5, minilm). Use when multiple final dirs exist.")
    args = parser.parse_args()

    output_root = args.output or OUTPUT

    registry = build_csv_registry(reranker=args.reranker)

    print(f"\n{'═' * 60}")
    print(f"  Thesis Plot Generator")
    print(f"  Output: {output_root}")
    print(f"  DPI: {args.dpi}")
    if args.reranker:
        print(f"  CLI Reranker Filter: {args.reranker}")
    print(f"{'═' * 60}\n")

    # Show what was found + metadata summary
    print("Found CSVs:")
    for exp, datasets in sorted(registry.items()):
        print(f"  {exp}:")
        for ds, entries in sorted(datasets.items()):
            for entry in entries:
                meta = entry.get("metadata", {})
                rr_pretty = _reranker_pretty_from_metadata(meta) or "unknown"
                ds_meta = meta.get("dataset_name", "?")
                if "modes" in entry:
                    for mode, p in sorted(entry["modes"].items()):
                        print(f"    {ds}/{mode}: {os.path.relpath(p, BASE)}  [reranker={rr_pretty}, dataset={ds_meta}]")
                else:
                    print(f"    {ds}: {os.path.relpath(entry['csv'], BASE)}  [reranker={rr_pretty}, dataset={ds_meta}]")
    print()

    failed = []  # Track failures

    # ── Cross-experiment comparisons ──
    if args.exp is None or args.exp.startswith("cross_"):
        cross_to_plot = list(CROSS_EXPERIMENT_PLOTTERS.keys())
        if args.exp:
            cross_to_plot = [e for e in cross_to_plot if e == args.exp]

        for cross_name in cross_to_plot:
            cross_plotter = CROSS_EXPERIMENT_PLOTTERS[cross_name]
            print(f"\n{'─' * 40}")
            print(f"Cross-experiment: {cross_name}")
            print(f"{'─' * 40}")
            try:
                cross_plotter(registry, output_root, args.dpi)
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append(f"{cross_name}: {e}")
                import traceback
                traceback.print_exc()

        # If only cross-experiment requested, skip per-experiment loop
        if args.exp and args.exp.startswith("cross_"):
            if failed:
                print(f"\n{'!' * 60}")
                print(f"  FAILURES ({len(failed)}):")
                for f in failed:
                    print(f"    ✗ {f}")
                print(f"{'!' * 60}")
            print(f"\n{'═' * 60}")
            print(f"  Done! All plots in: {output_root}")
            print(f"{'═' * 60}")
            return

    # Filter if --exp specified
    exps_to_plot = list(registry.keys())
    if args.exp:
        exps_to_plot = [e for e in exps_to_plot if e == args.exp]
        if not exps_to_plot:
            all_available = list(registry.keys()) + list(CROSS_EXPERIMENT_PLOTTERS.keys())
            print(f"ERROR: Experiment '{args.exp}' not found. Available: {all_available}")
            sys.exit(1)



    for exp_name in sorted(exps_to_plot):
        plotter = EXPERIMENT_PLOTTERS.get(exp_name)
        if plotter is None:
            print(f"SKIP: No plotter for '{exp_name}'")
            continue

        datasets = registry[exp_name]
        print(f"\n{'─' * 40}")
        print(f"Plotting {exp_name}")
        print(f"{'─' * 40}")

        for ds_pretty, entries in sorted(datasets.items()):
            for entry in entries:
                meta = entry.get("metadata", {})
                rr_pretty = _reranker_pretty_from_metadata(meta)

                # Update label map from metadata (not CLI)
                rr_type = meta.get("reranker_type")
                if rr_type:
                    _update_label_map(rr_type, exp_name=exp_name)

                # Validate: if --reranker given, check it matches metadata
                if args.reranker and rr_pretty:
                    cli_lower = args.reranker.lower()
                    meta_lower = (rr_pretty or "").lower()
                    rr_type_lower = (rr_type or "").lower()
                    if cli_lower not in meta_lower and cli_lower not in rr_type_lower:
                        msg = (f"{exp_name}/{ds_pretty}: CLI filter '{args.reranker}' "
                               f"does not match metadata reranker '{rr_pretty}' "
                               f"(type={rr_type}). SKIPPING.")
                        print(f"  ⚠ MISMATCH: {msg}")
                        failed.append(msg)
                        continue

                # Build output dir: always include reranker from metadata
                rr_dir = rr_pretty.lower().replace(" ", "_").replace("-", "") if rr_pretty else "unknown_reranker"
                out_dir = os.path.join(output_root, exp_name, ds_pretty, rr_dir)

                print(f"\n  Dataset: {ds_pretty}")
                print(f"  Reranker (from metadata): {rr_pretty or 'NOT FOUND'}")
                if not meta:
                    print(f"  ⚠ WARNING: No metadata found — reranker labels may be wrong")
                    failed.append(f"{exp_name}/{ds_pretty}: No metadata JSON found")

                # Print CSV paths
                if "modes" in entry:
                    for m, p in sorted(entry["modes"].items()):
                        print(f"  CSV ({m}): {os.path.relpath(p, BASE)}")
                else:
                    print(f"  CSV: {os.path.relpath(entry['csv'], BASE)}")
                print(f"  Output: {os.path.relpath(out_dir, BASE)}")

                try:
                    if "modes" in entry:
                        plotter(entry["modes"], out_dir, args.dpi, ds_pretty,
                                reranker_name=rr_pretty or "")
                    else:
                        plotter(entry["csv"], out_dir, args.dpi, ds_pretty,
                                reranker_name=rr_pretty or "")
                except Exception as e:
                    msg = f"{exp_name}/{ds_pretty}/{rr_pretty}: {e}"
                    print(f"  ERROR: {e}")
                    failed.append(msg)
                    import traceback
                    traceback.print_exc()

    # Summary
    if failed:
        print(f"\n{'!' * 60}")
        print(f"  FAILURES ({len(failed)}):")
        for f in failed:
            print(f"    ✗ {f}")
        print(f"{'!' * 60}")

    print(f"\n{'═' * 60}")
    print(f"  Done! All plots in: {output_root}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
