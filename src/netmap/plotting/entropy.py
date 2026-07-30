
import scanpy as sc
import pandas as pd
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt



import networkx as nx

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def compute_bcrank_metrics(bcrank_df, top_n=100):
    """Process a ranked gene DataFrame and compute Shannon entropy discovery metrics.

    Iterates through the top ``top_n`` genes in rank order, tracking which
    cell-type labels have been seen so far and computing the running Shannon
    entropy of the cell-type distribution at each rank. Also records the rank
    at which each new cell-type is first encountered.

    Args:
        bcrank_df (pd.DataFrame): Ranked gene DataFrame as returned by
            ``sc.get.rank_genes_groups_df``. Must have a ``'names'`` column
            containing strings in the format ``'<celltype>_<gene>'``. If a
            ``'celltype'`` column is not already present it is created by
            splitting ``'names'`` on the last underscore.
        top_n (int): Number of top-ranked genes to process. Defaults to 100.

    Returns:
        dict: Metrics dictionary with keys:

            - ``'ranks'`` (list of int): Rank indices 1 .. top_n.
            - ``'raw_h'`` (list of float): Running Shannon entropy at each rank.
            - ``'discoveries'`` (list of tuple): ``(rank, celltype)`` pairs
              recording the rank at which each cell-type was first seen.
            - ``'top_n'`` (int): The value of ``top_n`` used.
    """
    # 1. Preprocessing
    df = bcrank_df.head(top_n).copy()
    if 'celltype' not in df.columns:
        df[['celltype', 'gene']] = df['names'].str.rsplit('_', n=1, expand=True)

    ranked_terms = df['celltype'].tolist()

    # 2. Entropy and Discovery Calculation
    raw_h = []
    discoveries = []
    seen = set()
    counts = Counter()

    for i, term in enumerate(ranked_terms, 1):
        if term not in seen:
            discoveries.append((i, term)) # Store (Rank, Term Name)
            seen.add(term)

        counts[term] += 1
        probs = [count / i for count in counts.values()]

        # Shannon Entropy calculation
        h = -sum(p * np.log2(p) for p in probs if p > 0)
        raw_h.append(h)

    # Store results in a dictionary for easy passing to plotting
    metrics = {
        'ranks': list(range(1, len(ranked_terms) + 1)),
        'raw_h': raw_h,
        'discoveries': discoveries,
        'top_n': top_n
    }

    return metrics

def plot_bcrank_entropy(metrics, title='Cell-Type Discovery Entropy', figsize=(12, 7), label_limit=12):
    """Visualise the entropy discovery curve produced by :func:`compute_bcrank_metrics`.

    Plots the running Shannon entropy against gene rank, annotates the points at
    which each new cell-type is first discovered (up to ``label_limit``
    annotations), and applies consistent styling.

    Args:
        metrics (dict): Metrics dictionary as returned by
            :func:`compute_bcrank_metrics`, with keys ``'ranks'``, ``'raw_h'``,
            ``'discoveries'``, and ``'top_n'``.
        title (str): Base title for the plot. The value of ``top_n`` from
            ``metrics`` is appended automatically. Defaults to
            ``'Cell-Type Discovery Entropy'``.
        figsize (tuple): Figure size passed to ``plt.subplots``. Defaults to
            ``(12, 7)``.
        label_limit (int): Maximum number of discovery annotations to draw.
            Defaults to 12.

    Returns:
        None: Displays the figure via ``plt.show()``.
    """
    ranks = metrics['ranks']
    raw_h = metrics['raw_h']
    discoveries = metrics['discoveries']

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Plotting the curve
    ax.plot(ranks, raw_h, linewidth=2.5, color='#2c7bb6', label='Raw Entropy (H)', zorder=2)

    # Styling
    ax.set_xlabel('Gene Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shannon Entropy (bits)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotations
    for i, (rank, term) in enumerate(discoveries[:label_limit]):
        y_pos = raw_h[rank-1]
        ax.scatter(rank, y_pos, color='firebrick', s=50, edgecolors='white', zorder=5)

        # Vertical stagger logic
        v_offset = 30 + (i % 3 * 15)

        ax.annotate(
            term,
            xy=(rank, y_pos),
            xytext=(0, v_offset),
            textcoords='offset points',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8, connectionstyle="arc3,rad=0.1"),
            ha='center', fontsize=9, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8, lw=0.5)
        )

    if raw_h:
        ax.set_ylim(0, max(raw_h) * 1.4)

    ax.set_title(f"{title} (Top {metrics['top_n']} Genes)", fontsize=14, pad=15, fontweight='bold')
    plt.show()

def bcrank_entropy(bcrank_df, top_n=100, label_limit=12, return_metrics=False, **plot_kwargs):
    """Compute and plot cell-type discovery entropy in a single call.

    Convenience wrapper that calls :func:`compute_bcrank_metrics` followed by
    :func:`plot_bcrank_entropy`. Optionally returns the metrics dictionary for
    downstream use.

    Args:
        bcrank_df (pd.DataFrame): Ranked gene DataFrame as returned by
            ``sc.get.rank_genes_groups_df``. See :func:`compute_bcrank_metrics`
            for column requirements.
        top_n (int): Number of top-ranked genes to process. Defaults to 100.
        label_limit (int): Maximum number of discovery annotations on the plot.
            Defaults to 12.
        return_metrics (bool): If ``True``, return the metrics dict after plotting.
            Defaults to ``False``.
        **plot_kwargs: Additional keyword arguments forwarded to
            :func:`plot_bcrank_entropy` (e.g. ``title``, ``figsize``).

    Returns:
        dict or None: The metrics dictionary from :func:`compute_bcrank_metrics`
        when ``return_metrics=True``, otherwise ``None``.
    """
    # Run calculation
    metrics = compute_bcrank_metrics(bcrank_df, top_n=top_n)

    # Run plotting
    plot_bcrank_entropy(metrics, label_limit=label_limit, **plot_kwargs)

    if return_metrics:
        return metrics


