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


def rank_regulon_groups_dotplot(
    grn_adata_filtered,
    adata_regl,
    original_cluster_column='leiden',
    new_cluster_column='leiden_remap',
    n_genes=10,
    key="wilcoxon",
    cmap='bwr',
    figsize=(25, 2),
    values_to_plot="scores",
    return_fig=True,
    var_group_rotation=0,
    regulon_name_col=None,
):
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
            in ``.var`` (``{cluster}_nonzero`` pattern) and a ``source`` column
            identifying the regulator/TF for each row.
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
        regulon_name_col (str or None): Column in ``adata_regl.var`` holding the
            regulon identifier used to derive the TF/"source" (via splitting on
            ``'_'``). If ``None`` (default), the regulon identifier is taken from
            ``adata_regl.var_names`` (the index), matching the previous behaviour.
            Set this when the regulon name lives in a separate column instead of
            the index (e.g. the index is a numeric/arbitrary ID). Regardless of
            this setting, the fractions table is always keyed by
            ``adata_regl.var_names``, since that is what Scanpy uses as the
            dotplot's gene/regulon column labels.

    Returns:
        scanpy.plotting._dotplot.DotPlot or None: The dotplot object if
        ``return_fig=True``, else ``None``.
    """

    cluster_mapping = grn_adata_filtered.obs[[original_cluster_column, new_cluster_column]].drop_duplicates()
    leiden_to_celltype = dict(zip(cluster_mapping[original_cluster_column], cluster_mapping[new_cluster_column]))

    colheaders = grn_adata_filtered.var.columns[grn_adata_filtered.var.columns.str.contains('nonzero')]
    colheaders = [c for c in colheaders if c not in ('count_nonzero', 'count_nonzero_norm')]

    # Identifier used to derive the "source" TF for each regulon: either the
    # var index (default, previous behaviour) or an explicit column, for cases
    # where the human-readable regulon name is stored separately from the index.
    if regulon_name_col is not None:
        regulon_labels = adata_regl.var[regulon_name_col].astype(str)
    else:
        regulon_labels = pd.Series(adata_regl.var.index.astype(str), index=adata_regl.var.index)

    adata_regl.var['source'] = regulon_labels.str.split('_').str[-1].to_numpy()
    adata_regl.var['celltype'] = regulon_labels.str.split('_').str[0].to_numpy()


    # Fractions are always keyed by adata_regl.var_names, since that's what
    # Scanpy's dotplot uses for dot_color_df / dot_size_df column labels.
    fractions = {}
    for var_name, sou in zip(adata_regl.var_names, adata_regl.var['source']):
        mask = grn_adata_filtered.var['source'] == sou
        fractions[var_name] = grn_adata_filtered.var.loc[mask, colheaders].sum()

    fractions = pd.DataFrame(fractions)
    fractions.index = fractions.index.str.replace('_nonzero', '', regex=False)

    # return fig needs to be true: get plot, modify sizes, then plot or return
    pp = sc.pl.rank_genes_groups_dotplot(
        adata_regl,
        n_genes=n_genes,
        key=key,
        groupby=new_cluster_column,
        cmap=cmap,
        figsize=figsize,
        values_to_plot=values_to_plot,
        return_fig=True,
        var_group_rotation=var_group_rotation,
    )

    fractions = fractions.reindex(list(pp.dot_size_df.index))

    dot_size_df = fractions.loc[:, pp.dot_color_df.columns].astype(float)
    dot_size_df = dot_size_df.div(dot_size_df.max(axis=0), axis='columns')
    # Cast/fill explicitly before fillna to avoid the pandas "silent
    # downcasting on fillna" FutureWarning/behaviour change in newer pandas.
    pp.dot_size_df = dot_size_df.fillna(0.0)

    if return_fig:
        return pp
    else:
        pp.show()