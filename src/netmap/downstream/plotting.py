"""Regulon and GRN network visualisation utilities.

Wraps Scanpy dotplots with edge-enrichment overlays and provides NetworkX-based
inset network drawing for comparative regulon plots.
"""

import scanpy as sc
import pandas as pd
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt



import networkx as nx

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def rank_regulon_groups_dotplot(grn_adata_filtered, adata_regl, original_cluster_column='leiden', new_cluster_column='leiden_remap', n_genes=10, key="wilcoxon", cmap='bwr', figsize=(25, 2), values_to_plot="scores", return_fig=True, var_group_rotation=0):
    """Render a Scanpy dotplot overlaid with edge neighbourhood-support fractions.

    For each regulon shown in the dotplot, replaces the dot-size encoding with
    the normalised sum of per-cluster ``{cluster}_nonzero`` mask counts, giving a
    visual indication of co-expression support. The dot colour still encodes the
    Wilcoxon score from ``sc.tl.rank_genes_groups``.

    Note:
        Raises a Scanpy error if ``original_cluster_column`` and
        ``new_cluster_column`` map to different cell sets.

    Args:
        grn_adata_filtered (anndata.AnnData): GRN AnnData with mask-count columns
            in ``.var`` (``{cluster}_nonzero`` pattern).
        adata_regl (anndata.AnnData): AnnData of regulon UCell scores (obs=cells,
            var=regulons); must have ``rank_genes_groups`` result under ``key``.
        original_cluster_column (str): Cluster column matching the rank_genes result.
            Defaults to ``'leiden'``.
        new_cluster_column (str): Remapped cluster column used for the dotplot
            groupby axis. Defaults to ``'leiden_remap'``.
        n_genes (int): Number of top regulons per cluster to display. Defaults to 10.
        key (str): Key for ``rank_genes_groups`` result. Defaults to ``'wilcoxon'``.
        cmap (str): Colormap for dot colour. Defaults to ``'bwr'``.
        figsize (tuple): Figure size. Defaults to ``(25, 2)``.
        values_to_plot (str): Scanpy dotplot value encoding. Defaults to ``'scores'``.
        return_fig (bool): If ``True`` return the figure object; otherwise call
            ``.show()``. Defaults to ``True``.
        var_group_rotation (int): Rotation of var group labels. Defaults to 0.

    Returns:
        scanpy.plotting._dotplot.DotPlot or None: The dotplot object if
        ``return_fig=True``, else ``None``.
    """

    cluster_mapping = grn_adata_filtered.obs[[original_cluster_column, new_cluster_column]].drop_duplicates()
    leiden_to_celltype = dict(zip(cluster_mapping[original_cluster_column], cluster_mapping[new_cluster_column]))
    colheaders = grn_adata_filtered.var.columns[grn_adata_filtered.var.columns.str.contains('nonzero')]

    colheaders = list(colheaders)
    if 'count_nonzero' in colheaders:
        colheaders.remove('count_nonzero')
    if 'count_nonzero_norm' in colheaders:
        colheaders.remove('count_nonzero_norm')

    adata_regl.var['regulon_name'] = [x.replace('_UCell', '') for x in adata_regl.var.index]
    adata_regl.var['source'] = [x.split('_')[-1] for x in adata_regl.var['regulon_name']]

    fractions = {}
    count = 0
    for ri in range(len(adata_regl.var.regulon_name)):
        regulon = adata_regl.var.regulon_name[ri]
        sou = adata_regl.var.source[ri]
        count = count+1
        fractions[f'{regulon}'] =   grn_adata_filtered.var[grn_adata_filtered.var.source == sou][colheaders].sum()

    fractions = pd.DataFrame(fractions)
    fractions.index = [x.replace('_nonzero', '') for x in fractions.index]

    # return fig needs to be true: get plot, modify sizes, then plot or return
    pp = sc.pl.rank_genes_groups_dotplot(adata_regl, n_genes=n_genes, key=key, groupby=new_cluster_column, cmap=cmap, figsize=figsize, values_to_plot=values_to_plot, return_fig=True, var_group_rotation=var_group_rotation)
    print(pp)
    fractions = fractions.reindex(list(pp.dot_size_df.index))

    pp.dot_size_df = fractions.loc[:, pp.dot_color_df.columns]
    pp.dot_size_df = pp.dot_size_df/(pp.dot_size_df.max())
    pp.dot_size_df = pp.dot_size_df.fillna(0)

    if return_fig:
        return pp
    else:
        pp.show()




def get_grn_from_regulon(regulon_df, full_name, top_n=20):
    """Build a directed NetworkX graph for a single named regulon.

    Parses ``full_name`` of the form ``'<cluster>_<TF>'`` (split from the right
    to handle cluster names that contain underscores), filters the regulon table
    to the unique edges for that cluster and source, and constructs a directed
    graph from the top ``top_n`` edges ranked by ``sum_of_edge``.

    Args:
        regulon_df (pd.DataFrame): Flat regulon DataFrame with columns
            ``set_type``, ``cluster``, ``source``, ``target``, and ``sum_of_edge``,
            as produced by :func:`~netmap.downstream.regulon.make_cluster_regulon_dataframe`.
        full_name (str): Regulon identifier of the form ``'<cluster>_<TF>'``,
            e.g. ``'nk_cells_NKG7'``.
        top_n (int): Number of top edges (by ``sum_of_edge``) to include in the
            graph. Defaults to 20.

    Returns:
        networkx.DiGraph: Directed graph with source TF and target genes as nodes
        and edges corresponding to the top-ranked GRN edges.
    """
    # Split from the right to handle cluster names with underscores (e.g., cd8+_tcells)
    cluster, source = full_name.rsplit('_', 1)

    subset = regulon_df[
        (regulon_df.set_type == 'unique') &
        (regulon_df.cluster == cluster) &
        (regulon_df.source == source)
    ].nlargest(top_n, 'sum_of_edge')

    return nx.from_pandas_edgelist(subset, 'source', 'target', create_using=nx.DiGraph())

def draw_inset_graph(parent_ax, G, orientation='x'):
    """Draw a NetworkX graph as an inset axis attached to a parent matplotlib axes.

    Positions the inset below (``orientation='x'``) or to the left
    (``orientation='y'``) of the parent axes, using Kamada–Kawai layout for
    node placement. Graphs with three or fewer nodes use fixed axis limits to
    prevent edges from rendering as infinite lines.

    Args:
        parent_ax (matplotlib.axes.Axes): The axes to which the inset is
            attached.
        G (networkx.Graph): The graph to draw. An empty graph produces an
            invisible inset axes.
        orientation (str): ``'x'`` places the inset below the parent (x-axis
            label side); ``'y'`` places it to the left (y-axis label side).
            Defaults to ``'x'``.

    Returns:
        None
    """
    # Pushing the bbox further (to -0.5) to avoid any overlap with the axis frame
    if orientation == 'x':
        bbox = (0.0, -0.52, 1.0, 0.35)
    else:
        bbox = (-0.52, 0.0, 0.35, 1.0)

    ax_ins = inset_axes(parent_ax, width="100%", height="100%", loc='center',
                        bbox_to_anchor=bbox, bbox_transform=parent_ax.transAxes, borderpad=0)

    if len(G) > 0:
        pos = nx.kamada_kawai_layout(G)

        # Logic for 2-node graphs to keep edges from looking like infinite lines
        if len(G) <= 3:
            ax_ins.set_xlim(-2.5, 2.5)
            ax_ins.set_ylim(-2.5, 2.5)
        else:
            x_values, y_values = zip(*pos.values())
            x_r, y_r = max(x_values) - min(x_values), max(y_values) - min(y_values)
            ax_ins.set_xlim(min(x_values) - x_r*0.4, max(x_values) + x_r*0.4)
            ax_ins.set_ylim(min(y_values) - y_r*0.4, max(y_values) + y_r*0.4)

        nx.draw_networkx_edges(G, pos, ax=ax_ins, edge_color='#bdc3c7', alpha=0.4, width=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax_ins, node_size=100, node_color='#f8f9fa',
                               edgecolors='#34495e', linewidths=0.5)
        nx.draw_networkx_labels(G, pos, ax=ax_ins, font_size=7, font_weight='bold', clip_on=False)

    ax_ins.axis('off')

def plot_regulon_comparison(adata, regulon_table, regulons, cluster_key='leiden_remap', palette=None, show_legend=True):
    """Plot a scatter comparison of two regulon activity scores with inset GRN graphs.

    Creates a scatter plot of the activity scores for two regulons (columns of
    ``adata``) with cells coloured by cluster. Inset NetworkX graphs drawn below
    (x-axis regulon) and to the left (y-axis regulon) show the underlying GRN
    structure for each regulon.

    Args:
        adata (anndata.AnnData): AnnData whose ``.X`` columns correspond to
            regulon activity scores and whose ``obs[cluster_key]`` holds cluster
            labels. The two regulon names in ``regulons`` must be present in
            ``adata.var_names``.
        regulon_table (pd.DataFrame): Flat regulon DataFrame with columns
            ``set_type``, ``cluster``, ``source``, ``target``, and ``sum_of_edge``,
            as produced by
            :func:`~netmap.downstream.regulon.make_cluster_regulon_dataframe`.
        regulons (list of str): Exactly two regulon names; ``regulons[0]`` is
            plotted on the x-axis and ``regulons[1]`` on the y-axis.
        cluster_key (str): Column in ``adata.obs`` used for colour grouping.
            Defaults to ``'leiden_remap'``.
        palette (dict or None): Colour palette passed to ``seaborn.scatterplot``.
            Defaults to ``None`` (seaborn default).
        show_legend (bool): If ``True``, display the cluster legend. Defaults to
            ``True``.

    Returns:
        tuple: ``(fig, ax)`` — the matplotlib Figure and Axes objects.
    """
    # 1. Prepare Scatter Data
    df = pd.DataFrame(adata[:, regulons].X.copy(), columns=regulons)
    df['group'] = adata.obs[cluster_key].values

    # 2. Build Graphs
    G_x = get_grn_from_regulon(regulon_table, regulons[0])
    G_y = get_grn_from_regulon(regulon_table, regulons[1])

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(4, 4))

    # Increase margins significantly to accommodate external GRNs and titles
    plt.subplots_adjust(top=0.82, right=0.82, left=0.18, bottom=0.18)

    sns.scatterplot(
        data=df, x=regulons[0], y=regulons[1], hue='group',
        palette=palette, ax=ax, s=20, alpha=0.5, legend=show_legend
    )

    # Draw Insets with extra clearance
    draw_inset_graph(ax, G_x, 'x')
    draw_inset_graph(ax, G_y, 'y')

    # 4. Move Labels and Shrink Ticks
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')

    # Shrink the numbers on the axes
    ax.tick_params(axis='both', which='major', labelsize=7)

    # 5. Legend Styling
    if show_legend:
        ax.legend(
            title=cluster_key, bbox_to_anchor=(1.2, 1), loc='upper left',
            fontsize=6, title_fontsize=7, frameon=False
        )
    elif ax.get_legend():
        ax.get_legend().remove()

    sns.despine(ax=ax)
    return fig, ax


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
