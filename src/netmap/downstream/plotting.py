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



def _resolve_celltypes(celltype_labels, celltypes, celltype_col):
    """Validate and normalize a ``celltypes`` filter against the labels actually present.

    Returns:
        list: The celltypes to iterate over — every unique label in
        ``celltype_labels`` if ``celltypes`` is ``None``, else the requested
        subset (after checking every one of them exists).
    """
    available = pd.unique(celltype_labels)
    if celltypes is None:
        return list(available)
    selected = [celltypes] if isinstance(celltypes, str) else list(celltypes)
    missing = set(selected) - set(available)
    if missing:
        raise ValueError(
            f"Celltype(s) {sorted(missing)} not found in grn_adata.obs['{celltype_col}']. "
            f"Available: {sorted(available)}"
        )
    return selected


def _masked_edge_strength(grn_adata, cell_idx, edge_col_idx):
    """Sum |masked attribution| over a set of cells, for a set of edge columns.

    "Masked" means elementwise-multiplied by ``grn_adata.layers['mask']`` —
    zeroing out edges not co-expressed in a cell's kNN neighbourhood — before
    summing, so an edge only accrues strength from cells where it's actually
    supported, not just theoretically present in the GRN. Mirrors the
    ``np.multiply(grn_adata.X, grn_adata.layers['mask'])`` masking step in
    :func:`~netmap.downstream.celltype_random_walk.run_celltype_random_walk`.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData with a populated ``X`` and a
            ``layers['mask']`` of matching shape.
        cell_idx (np.ndarray): Integer row indices selecting the cells to sum
            over.
        edge_col_idx (np.ndarray): Integer column indices selecting the edges
            to compute strength for.

    Returns:
        np.ndarray: Summed absolute masked attribution per edge, shape
        ``(len(edge_col_idx),)``.
    """
    x = grn_adata.X[cell_idx][:, edge_col_idx]
    mask = grn_adata.layers["mask"][cell_idx][:, edge_col_idx]
    masked = np.multiply(x, mask)
    masked_abs = abs(masked) if scs.issparse(masked) else np.abs(masked)
    return np.asarray(masked_abs.sum(axis=0)).flatten()


def _select_top_genes_for_celltype(top_genes_per_cell, cell_idx, top_n, max_nodes):
    """Count gene recurrence in the top-``top_n`` list across a celltype's cells.

    Returns:
        tuple: ``(gene_counts, top_genes)`` — a ``Counter`` of recurrence per
        gene, and the ``max_nodes`` most recurrent gene names.
    """
    gene_counts = Counter()
    for i in cell_idx:
        for gene, _score in top_genes_per_cell[i][:top_n]:
            gene_counts[gene] += 1
    top_genes = [gene for gene, _ in gene_counts.most_common(max_nodes)]
    return gene_counts, top_genes


def _select_top_edges_for_celltype(grn_adata, cell_idx, gene_set, top_edges):
    """Pick the ``top_edges`` strongest actually-expressed edges among ``gene_set``.

    "Candidate" edges are theoretical GRN edges (``grn_adata.var``) with both
    endpoints in ``gene_set``, self-loops excluded. Each candidate is scored by
    :func:`_masked_edge_strength` over ``cell_idx``, and only the top-scoring
    ``top_edges`` survive.

    Returns:
        pd.DataFrame: The kept edges (columns from ``grn_adata.var`` plus
        ``'strength'``), possibly empty.
    """
    candidate_edges = grn_adata.var[
        grn_adata.var["source"].isin(gene_set) & grn_adata.var["target"].isin(gene_set)
    ]
    candidate_edges = candidate_edges[candidate_edges["source"] != candidate_edges["target"]]
    if candidate_edges.empty:
        return candidate_edges.assign(strength=[])

    edge_col_idx = grn_adata.var.index.get_indexer(candidate_edges.index)
    strength = _masked_edge_strength(grn_adata, cell_idx, edge_col_idx)
    return candidate_edges.assign(strength=strength).nlargest(top_edges, "strength")


def _draw_celltype_graph(ct, top_genes, gene_counts, kept_edges, n_cells_ct, min_node_size, max_node_size, figsize):
    """Build the igraph.Graph for one celltype and render it to a new figure.

    Returns:
        tuple: ``(fig, graph)``.
    """
    graph = ig.Graph(directed=True)
    graph.add_vertices(top_genes)
    if not kept_edges.empty:
        graph.add_edges(list(zip(kept_edges["source"], kept_edges["target"])))
        graph.es["strength"] = kept_edges["strength"].tolist()

    counts = np.array([gene_counts[name] for name in graph.vs["name"]], dtype=float)
    fraction = counts / n_cells_ct
    sizes = min_node_size + fraction * (max_node_size - min_node_size)
    graph.vs["size"] = sizes.tolist()
    graph.vs["label"] = graph.vs["name"]
    graph.vs["top_count"] = counts.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    layout = graph.layout("fr")
    ig.plot(
        graph,
        target=ax,
        layout=layout,
        vertex_size=graph.vs["size"],
        vertex_label=graph.vs["label"],
        vertex_label_size=7,
        edge_arrow_size=0.5,
        edge_color="#bdc3c7",
    )
    ax.set_title(f"{ct} (top {len(top_genes)} genes)")
    return fig, graph


def plot_celltype_top_gene_networks(
    grn_adata,
    top_genes_per_cell,
    celltype_col="celltype",
    celltypes=None,
    top_n=20,
    max_nodes=100,
    top_edges=20,
    min_node_size=10,
    max_node_size=60,
    figsize=(8, 8),
    out_path=None,
):
    """Draw one GRN-edge network per celltype, sized by top-20 recurrence.

    For each celltype (grouped by ``grn_adata.obs[celltype_col]``), counts how
    many of that celltype's cells have each gene in their top-``top_n`` RWR
    genes (as returned by
    :func:`~netmap.downstream.celltype_random_walk.run_celltype_random_walk`
    with ``top_genes_k`` set), and keeps at most ``max_nodes`` genes by that
    count as nodes.

    Edges are not the full theoretical GRN — among candidate edges connecting
    two kept genes, each edge's strength is the summed absolute *masked*
    attribution (``grn_adata.X * grn_adata.layers['mask']``) over that
    celltype's own cells (see :func:`_masked_edge_strength`), and only the
    ``top_edges`` strongest actually-expressed edges are drawn. Self-loops are
    dropped. Genes with none of their edges among the kept ``top_edges`` still
    appear as isolated nodes.

    Node size is a fraction of that celltype's own cell count — a gene in the
    top-``top_n`` of every cell in its celltype gets ``max_node_size``, one
    that never appears gets ``min_node_size`` — so sizes stay comparable
    across celltypes with different numbers of cells.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData with ``source``/``target``
            columns in ``.var``, a celltype label in ``.obs``, a populated
            ``X``, and a ``layers['mask']`` of matching shape.
        top_genes_per_cell (list): One entry per cell, each a list of
            ``(gene_name, score)`` tuples sorted descending by score, as
            returned by ``run_celltype_random_walk(..., top_genes_k=...)``.
            Must be in the same cell order as ``grn_adata.obs``, and each
            entry must contain at least ``top_n`` genes.
        celltype_col (str): Column in ``grn_adata.obs`` giving each cell's
            celltype label. Defaults to ``'celltype'``.
        celltypes (str or list of str, optional): Restrict plotting to this
            celltype (or list of celltypes), instead of every unique value in
            ``celltype_col``. Raises ``ValueError`` if any requested celltype
            isn't present. ``None`` (default) plots all of them.
        top_n (int): Number of top genes per cell to count towards
            recurrence. Must be ``<= len(top_genes_per_cell[i])``. Defaults
            to 20.
        max_nodes (int): Maximum number of genes (nodes) kept per celltype,
            by descending recurrence count. Defaults to 100.
        top_edges (int): Maximum number of edges kept per celltype, by
            descending masked-attribution strength (summed over that
            celltype's cells). Defaults to 20.
        min_node_size (float): Node size for a gene with 0% recurrence in its
            celltype. Defaults to 10.
        max_node_size (float): Node size for a gene at 100% recurrence in its
            celltype. Defaults to 60.
        figsize (tuple): Figure size per celltype plot. Defaults to ``(8, 8)``.
        out_path (str, optional): Directory to save each celltype's figure to,
            as ``'{out_path}/{celltype}_top_genes_network.png'``. ``None``
            (default) skips saving — figures are only returned.

    Returns:
        dict: ``{celltype: (fig, igraph.Graph)}`` — the matplotlib figure and
        underlying igraph graph (with ``'name'``, ``'size'``, and
        ``'top_count'`` vertex attributes, and per-edge ``'strength'``) for
        each celltype.
    """
    celltype_labels = grn_adata.obs[celltype_col].to_numpy()
    if len(top_genes_per_cell) != len(celltype_labels):
        raise ValueError(
            f"top_genes_per_cell has {len(top_genes_per_cell)} entries but "
            f"grn_adata has {len(celltype_labels)} cells."
        )

    selected_celltypes = _resolve_celltypes(celltype_labels, celltypes, celltype_col)

    results = {}
    for ct in selected_celltypes:
        cell_idx = np.nonzero(celltype_labels == ct)[0]

        gene_counts, top_genes = _select_top_genes_for_celltype(
            top_genes_per_cell, cell_idx, top_n, max_nodes
        )
        kept_edges = _select_top_edges_for_celltype(
            grn_adata, cell_idx, set(top_genes), top_edges
        )
        fig, graph = _draw_celltype_graph(
            ct, top_genes, gene_counts, kept_edges, len(cell_idx),
            min_node_size, max_node_size, figsize,
        )

        if out_path is not None:
            fig.savefig(f"{out_path}/{ct}_top_genes_network.png", dpi=150, bbox_inches="tight")

        results[ct] = (fig, graph)

    return results


def _gather_combined_celltype_data(
    grn_adata, top_genes_per_cell, celltype_labels, selected_celltypes,
    top_n, max_nodes_per_celltype, top_edges_per_celltype,
):
    """Run per-celltype gene/edge selection and merge into combined lookups.

    Returns:
        tuple:
            - gene_fractions (dict): ``{gene: {celltype: fraction}}`` — for
              each celltype where the gene was among that celltype's kept top
              genes, the fraction of that celltype's cells having it in their
              top-``top_n`` list.
            - combined_edges (dict): ``{(source, target): {'n_celltypes': int,
              'strength': float}}`` — merged across every celltype whose kept
              top edges included that pair.
    """
    gene_fractions = {}
    combined_edges = {}

    for ct in selected_celltypes:
        cell_idx = np.nonzero(celltype_labels == ct)[0]
        n_cells_ct = len(cell_idx)

        gene_counts, top_genes = _select_top_genes_for_celltype(
            top_genes_per_cell, cell_idx, top_n, max_nodes_per_celltype
        )
        for gene in top_genes:
            gene_fractions.setdefault(gene, {})[ct] = gene_counts[gene] / n_cells_ct

        kept_edges = _select_top_edges_for_celltype(
            grn_adata, cell_idx, set(top_genes), top_edges_per_celltype
        )
        for source, target, strength in zip(
            kept_edges["source"], kept_edges["target"], kept_edges["strength"]
        ):
            entry = combined_edges.setdefault((source, target), {"n_celltypes": 0, "strength": 0.0})
            entry["n_celltypes"] += 1
            entry["strength"] += strength

    return gene_fractions, combined_edges


def _layout_with_isolates_on_rim(graph, seed, k=None):
    """Force-directed layout for the connected part; isolated genes placed on a rim.

    Spring layout's repulsion has nothing to balance an isolated node against
    (no edge pulls it back in), so isolates tend to drift arbitrarily far out.
    That inflates the plot's bounding box with mostly-empty margin while the
    actual connected component stays small and crowded near the centre.
    Instead, only the non-isolated nodes go through ``spring_layout``; isolates
    are placed evenly on a circle just outside that component's extent, so the
    bounding box stays tight and the connected core gets the full ``k``-driven
    spread to itself.

    Args:
        graph (networkx.DiGraph): The full graph, isolates included.
        seed (int): Random seed for ``spring_layout``.
        k (float, optional): Target distance between nodes, passed straight to
            ``spring_layout`` — larger spreads nodes further apart, smaller
            pulls them tighter. ``None`` (default) auto-scales as
            ``3.0 / sqrt(n_connected_nodes)``.

    Returns:
        dict: ``{node: (x, y)}`` for every node in ``graph``.
    """
    isolates = list(nx.isolates(graph))
    connected_nodes = [n for n in graph.nodes if n not in isolates]

    if not connected_nodes:
        angles = np.linspace(0, 2 * np.pi, len(isolates), endpoint=False)
        return {node: (np.cos(a), np.sin(a)) for node, a in zip(isolates, angles)}

    subgraph = graph.subgraph(connected_nodes)
    if k is None:
        k = 3.0 / np.sqrt(len(connected_nodes))
    pos = nx.spring_layout(subgraph, k=k, iterations=200, seed=seed)

    if not isolates:
        return pos

    center = np.mean(np.array(list(pos.values())), axis=0)
    max_radius = max(np.linalg.norm(np.array(p) - center) for p in pos.values())
    rim_radius = max_radius * 1.25 + 0.1

    angles = np.linspace(0, 2 * np.pi, len(isolates), endpoint=False)
    for node, angle in zip(isolates, angles):
        pos[node] = center + rim_radius * np.array([np.cos(angle), np.sin(angle)])

    return pos


def compute_combined_celltype_network(
    grn_adata,
    top_genes_per_cell,
    celltype_col="celltype",
    celltypes=None,
    top_n=20,
    max_nodes_per_celltype=100,
    top_edges_per_celltype=20,
    max_total_nodes=100,
):
    """Select genes/edges across celltypes and build the combined network graph.

    Runs the same per-celltype selection as :func:`plot_celltype_top_gene_networks`
    (top-``top_n`` gene recurrence, ``top_edges_per_celltype`` strongest
    masked-attribution edges per celltype — see :func:`_masked_edge_strength`),
    then merges the results into one graph:

    - Every gene selected by at least one celltype is a candidate node; the
      ``max_total_nodes`` with the highest summed recurrence fraction (across
      all celltypes they were selected in) are kept.
    - Every edge kept by at least one celltype (with both endpoints surviving
      the node cap) is kept once, annotated with how many celltypes supported
      it and their summed strength.

    Does no plotting — this is the potentially-expensive step (masked edge
    strength over every cell). Pass the result to
    :func:`plot_combined_celltype_network` to render it, as many times as you
    want with different layout/style parameters, without recomputing this.
    :func:`combined_celltype_network` does both steps in one call.

    Args:
        grn_adata (anndata.AnnData): GRN AnnData with ``source``/``target``
            columns in ``.var``, a celltype label in ``.obs``, a populated
            ``X``, and a ``layers['mask']`` of matching shape.
        top_genes_per_cell (list): One entry per cell, each a list of
            ``(gene_name, score)`` tuples sorted descending by score, as
            returned by ``run_celltype_random_walk(..., top_genes_k=...)``.
            Must be in the same cell order as ``grn_adata.obs``.
        celltype_col (str): Column in ``grn_adata.obs`` giving each cell's
            celltype label. Defaults to ``'celltype'``.
        celltypes (str or list of str, optional): Restrict to this celltype
            (or list of celltypes) instead of every unique value in
            ``celltype_col``. ``None`` (default) uses all of them.
        top_n (int): Number of top genes per cell counted towards recurrence.
            Defaults to 20.
        max_nodes_per_celltype (int): Per-celltype node cap passed to
            :func:`_select_top_genes_for_celltype`. Defaults to 100.
        top_edges_per_celltype (int): Per-celltype edge cap passed to
            :func:`_select_top_edges_for_celltype`. Defaults to 20.
        max_total_nodes (int): Maximum number of genes kept in the combined
            network, by descending summed recurrence fraction across
            celltypes. Defaults to 100.

    Returns:
        dict: with keys ``'graph'`` (networkx.DiGraph, nodes are the kept
        genes, edges the kept GRN edges), ``'gene_fractions'`` (``{gene:
        {celltype: fraction}}`` for the kept genes), ``'importance'`` (``{gene:
        summed fraction across celltypes}``), ``'edge_items'`` (list of
        ``((source, target), {'n_celltypes', 'strength'})`` for the kept
        edges), ``'celltypes'`` (the resolved celltype list), and
        ``'celltype_col'`` (for the legend title).
    """
    celltype_labels = grn_adata.obs[celltype_col].to_numpy()
    if len(top_genes_per_cell) != len(celltype_labels):
        raise ValueError(
            f"top_genes_per_cell has {len(top_genes_per_cell)} entries but "
            f"grn_adata has {len(celltype_labels)} cells."
        )
    selected_celltypes = _resolve_celltypes(celltype_labels, celltypes, celltype_col)

    gene_fractions, combined_edges = _gather_combined_celltype_data(
        grn_adata, top_genes_per_cell, celltype_labels, selected_celltypes,
        top_n, max_nodes_per_celltype, top_edges_per_celltype,
    )
    if not gene_fractions:
        raise ValueError("No genes were selected for any of the requested celltypes.")

    importance = {gene: sum(fracs.values()) for gene, fracs in gene_fractions.items()}
    kept_genes = sorted(importance, key=importance.get, reverse=True)[:max_total_nodes]
    kept_gene_set = set(kept_genes)
    kept_gene_fractions = {gene: gene_fractions[gene] for gene in kept_genes}

    kept_edge_items = [
        (pair, info) for pair, info in combined_edges.items()
        if pair[0] in kept_gene_set and pair[1] in kept_gene_set
    ]

    graph = nx.DiGraph()
    graph.add_nodes_from(kept_genes)
    graph.add_edges_from(pair for pair, _ in kept_edge_items)

    return {
        "graph": graph,
        "gene_fractions": kept_gene_fractions,
        "importance": importance,
        "edge_items": kept_edge_items,
        "celltypes": selected_celltypes,
        "celltype_col": celltype_col,
    }


def _drop_isolates(graph, gene_fractions, importance):
    """Remove genes with no edges to any other kept gene.

    Returns:
        tuple: ``(graph, gene_fractions, importance)`` filtered to exclude
        isolated genes. Returned as-is if there are none.
    """
    isolates = set(nx.isolates(graph))
    if not isolates:
        return graph, gene_fractions, importance

    keep = [n for n in graph.nodes if n not in isolates]
    filtered_graph = graph.subgraph(keep).copy()
    filtered_fractions = {gene: gene_fractions[gene] for gene in keep}
    filtered_importance = {gene: importance[gene] for gene in keep}
    return filtered_graph, filtered_fractions, filtered_importance


def _compute_layout(graph, layout, seed, k):
    """Compute 2D node positions for the combined network under different algorithms.

    ``'spring'`` is a force simulation: attraction along edges, repulsion
    everywhere else. It degrades on genuinely dense graphs (many edges per
    node) because the repulsion has too many competing attractions to push
    against — no choice of ``k`` fixes that, since the crowding is a property
    of the edge/node ratio, not the layout algorithm. The other options don't
    share that failure mode.

    Args:
        graph (networkx.DiGraph): The graph to lay out.
        layout (str): One of:

            - ``'spring'`` (default): ``spring_layout`` on the connected part,
              isolated genes placed on a rim (see
              :func:`_layout_with_isolates_on_rim`). Tunable via ``k``.
            - ``'kk'``: igraph's Kamada-Kawai — positions reflect
              graph-theoretic shortest-path distances rather than a physical
              simulation; tends to spread small-to-medium graphs more evenly.
            - ``'drl'``: igraph's DrL (Distributed Recursive Layout) —
              purpose-built for large/dense graphs, generally separates
              clusters better than spring or KK when there are many edges.
            - ``'circular'``: nodes evenly spaced on a circle. Guarantees zero
              node-node overlap regardless of density — edges may cross a
              lot, but every node stays legible.
        seed (int): Random seed, used only by ``'spring'``.
        k (float, optional): Target inter-node distance, used only by
            ``'spring'``.

    Returns:
        dict: ``{node: (x, y)}`` for every node in ``graph``.
    """
    if layout == "spring":
        return _layout_with_isolates_on_rim(graph, seed, k=k)
    if layout == "circular":
        return nx.circular_layout(graph)
    if layout not in ("kk", "drl"):
        raise ValueError(f"Unknown layout {layout!r}; choose 'spring', 'kk', 'drl', or 'circular'.")

    node_names = list(graph.nodes)
    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(node_names)
    ig_graph.add_edges(list(graph.edges))
    ig_pos = ig_graph.layout(layout)
    return {name: tuple(ig_pos[i]) for i, name in enumerate(node_names)}


def plot_combined_celltype_network(
    network,
    node_radius_frac=(0.015, 0.05),
    palette=None,
    figsize=(10, 10),
    seed=42,
    k=None,
    layout="spring",
    drop_isolates=False,
    out_path=None,
):
    """Render a network built by :func:`compute_combined_celltype_network`.

    Pure rendering — layout plus the per-gene pie charts. Safe to call
    repeatedly with different ``node_radius_frac``/``palette``/``seed``/``k``/
    ``layout``/``drop_isolates`` against the same precomputed ``network``
    without redoing the (expensive) per-celltype gene/edge selection.

    Args:
        network (dict): As returned by :func:`compute_combined_celltype_network`.
        node_radius_frac (tuple): ``(min, max)`` pie radius, each as a fraction
            of the layout's coordinate span — ``min`` for the least-recurrent
            kept gene, ``max`` for the most-recurrent. Defaults to
            ``(0.015, 0.05)``.
        palette (dict, optional): ``{celltype: color}`` mapping. ``None``
            (default) assigns colors from matplotlib's ``tab20`` colormap.
        figsize (tuple): Figure size. Defaults to ``(10, 10)``.
        seed (int): Random seed for the ``'spring'`` layout, for
            reproducibility. Defaults to 42. Unused by other layouts.
        k (float, optional): Target distance between nodes in the ``'spring'``
            layout — increase to spread nodes further apart, decrease to pull
            them tighter. ``None`` (default) auto-scales as
            ``3.0 / sqrt(n_connected_nodes)``. Unused by other layouts.
        layout (str): One of ``'spring'`` (default), ``'kk'``, ``'drl'``, or
            ``'circular'`` — see :func:`_compute_layout`. If the graph still
            looks crowded after trying these, the more fundamental fix is
            usually to lower ``top_edges_per_celltype`` (fewer edges per kept
            gene) rather than change the layout.
        drop_isolates (bool): If ``True``, genes with no edge to any other
            kept gene are excluded from the plot entirely instead of being
            placed on a rim. Defaults to ``False``.
        out_path (str, optional): Directory to save the figure to, as
            ``'{out_path}/combined_celltype_network.png'``. ``None`` (default)
            skips saving.

    Returns:
        tuple: ``(fig, ax)``.
    """
    graph = network["graph"]
    gene_fractions = network["gene_fractions"]
    importance = network["importance"]
    if drop_isolates:
        graph, gene_fractions, importance = _drop_isolates(graph, gene_fractions, importance)

    pos = _compute_layout(graph, layout, seed, k)
    fig, ax = _draw_combined_celltype_graph(
        graph, pos, gene_fractions, network["edge_items"], importance,
        network["celltypes"], node_radius_frac, palette, figsize, network["celltype_col"],
    )

    if out_path is not None:
        fig.savefig(f"{out_path}/combined_celltype_network.png", dpi=150, bbox_inches="tight")

    return fig, ax


def combined_celltype_network(
    grn_adata,
    top_genes_per_cell,
    celltype_col="celltype",
    celltypes=None,
    top_n=20,
    max_nodes_per_celltype=100,
    top_edges_per_celltype=20,
    max_total_nodes=100,
    **plot_kwargs,
):
    """Compute and render the combined celltype gene network in one call.

    Convenience wrapper around :func:`compute_combined_celltype_network` and
    :func:`plot_combined_celltype_network`. If you're tuning layout or style
    (``node_radius_frac``, ``palette``, ``seed``, ``k``), call those two
    directly instead and reuse the same computed network — this recomputes
    gene/edge selection on every call.

    Args:
        grn_adata, top_genes_per_cell, celltype_col, celltypes, top_n,
            max_nodes_per_celltype, top_edges_per_celltype, max_total_nodes:
            Forwarded to :func:`compute_combined_celltype_network`.
        **plot_kwargs: Forwarded to :func:`plot_combined_celltype_network`
            (``node_radius_frac``, ``palette``, ``figsize``, ``seed``, ``k``,
            ``out_path``).

    Returns:
        tuple: ``(fig, ax, network)`` — the built network is always returned
        alongside the plot, since computing it is unavoidable here anyway;
        reuse it with :func:`plot_combined_celltype_network` to re-render
        without recomputing.
    """
    network = compute_combined_celltype_network(
        grn_adata, top_genes_per_cell, celltype_col, celltypes,
        top_n, max_nodes_per_celltype, top_edges_per_celltype, max_total_nodes,
    )
    fig, ax = plot_combined_celltype_network(network, **plot_kwargs)
    return fig, ax, network


def _draw_combined_celltype_graph(
    graph, pos, gene_fractions, edge_items, importance, selected_celltypes,
    node_radius_frac, palette, figsize, celltype_col,
):
    """Render the combined graph: edges via networkx, nodes as per-gene pie charts.

    Returns:
        tuple: ``(fig, ax)``.
    """
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    min_radius = node_radius_frac[0] * span
    max_radius = node_radius_frac[1] * span
    max_importance = max(importance[g] for g in graph.nodes)

    if palette is None:
        cmap = plt.get_cmap("tab20")
        palette = {ct: cmap(i % 20) for i, ct in enumerate(selected_celltypes)}

    fig, ax = plt.subplots(figsize=figsize)

    if edge_items:
        widths = [0.5 + info["n_celltypes"] for _, info in edge_items]
        nx.draw_networkx_edges(
            graph, pos, ax=ax, edge_color="#bdc3c7", alpha=0.5, width=widths,
            arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.05",
        )

    for gene in graph.nodes:
        fracs = gene_fractions[gene]
        total = sum(fracs.values())
        sizes = [f / total for f in fracs.values()]
        colors = [palette[ct] for ct in fracs]
        radius = min_radius + (importance[gene] / max_importance) * (max_radius - min_radius)

        x, y = pos[gene]
        ax.pie(sizes, colors=colors, radius=radius, center=(x, y),
               wedgeprops={"linewidth": 0.3, "edgecolor": "white"})
        ax.text(x, y + radius * 1.4, gene, ha="center", va="bottom", fontsize=6)

    ax.set_xlim(min(xs) - span * 0.15, max(xs) + span * 0.15)
    ax.set_ylim(min(ys) - span * 0.15, max(ys) + span * 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Combined top-gene network ({len(graph.nodes)} genes, {graph.number_of_edges()} edges)")

    legend_handles = [Patch(color=palette[ct], label=ct) for ct in selected_celltypes]
    ax.legend(handles=legend_handles, title=celltype_col, loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)

    return fig, ax
