import scanpy as sc
import pandas as pd
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt



import networkx as nx

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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

